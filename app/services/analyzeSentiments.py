import pandas as pd
import openai
import json
import tiktoken
import time
import os
from dotenv import load_dotenv
from collections import Counter
import random
from concurrent.futures import ThreadPoolExecutor
from openai.error import RateLimitError, APIError, Timeout
import re
import reanalyze_failed_batches

# Load environment variables from the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = 1800
ENCODING = tiktoken.encoding_for_model(MODEL)

# Define the base directory for file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Create a directory for failed batches if it doesn't exist
# Note: Its implemented but not used, due to the time it takes to reanalyze the batches --> Not worth it.
FALLITS_PATH = os.path.join(BASE_DIR, "data", "LotesFallits")
os.makedirs(FALLITS_PATH, exist_ok=True)
FALLITS_FILE = os.path.join(FALLITS_PATH, "batches_fallits.jsonl")

# Function to load comments from a CSV file
def load_comments(csv_path):
    df = pd.read_csv(csv_path)
    comentaris = []
    for _, row in df.iterrows():
        text = str(row.get("Comment Text", "")).strip()
        if not text:
            continue
        comentaris.append({
            "text": text,
            "vots": int(row.get("Comment Score (Upvotes)", 0)),
            "respostes": int(row.get("Number of Replies", 0))
        })
    return comentaris

# Function to divide comments into batches without exceeding token limit
def divice_by_barches(comentaris):
    batches = []
    lot = []
    tokens_lot = 0
    for c in comentaris:
        tokens = len(ENCODING.encode(c["text"]))
        if tokens_lot + tokens > MAX_TOKENS:
            batches.append(lot)
            lot = []
            tokens_lot = 0
        lot.append(c)
        tokens_lot += tokens
    if lot:
        batches.append(lot)
    return batches

import re

# Analyze a single batch of comments using OpenAI API
def analitzar_batch(batch, topic):
    # Build the prompt with the topic and comments
    texts = [c["text"] for c in batch]
    prompt = f"""
Analyze the following Reddit comments about the topic \"{topic}\":

1. Classify each comment as 'positiu', 'negatiu', or 'neutre'.
2. Provide standalone, self-contained summaries of positive and negative opinions (avoid vague replies, partial phrases, or references to other comments).
3. Include the most relevant comments with high votes or replies.

Expected JSON format:
{{
  "sentiments": {{"positiu": 0, "negatiu": 0, "neutre": 0}},
  "opinions": {{
    "positives": [],
    "negatives": []
  }},
  "comentaris": [
    {{
      "text": "...",
      "sentiment": "...",
      "vots": ...,
      "respostes": ...
    }}
  ]
}}

Comments:
{texts}
"""

    try:
        # Send the request to the OpenAI API
        resposta = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = resposta.choices[0].message.content

        # Fix broken escape sequences
        content = re.sub(r'\\(?![nrt"\\/bfu])', r'\\\\', content)

        # Attempt to parse the response as JSON
        try:
            return json.loads(content)

        # Handle invalid JSON responses
        except json.JSONDecodeError as e:
            print("❌ JSON invàlid rebut (primeres línies):")
            print(content[:1000])  # Limit output to 1000 chars for debugging
            print("🔍 Error:", e)

            # If response looks incomplete or truncated, check for unclosed braces
            if not content.strip().endswith("}") or content.count("{") > content.count("}"):
                print("⛔️ Resposta probablement truncada. Lot guardat a fitxer.")

            # Save failed batch for later reanalysis
            with open(FALLITS_FILE, "a", encoding="utf-8") as f:
                json.dump({"topic": topic, "batch": batch}, f, ensure_ascii=False)
                f.write("\n")

            return None

    except Exception as e:
        # Handle general errors
        print("⚠️ Error greu en batch:", e)

        # Save the failed batch for later reanalysis
        with open(FALLITS_FILE, "a", encoding="utf-8") as f:
            json.dump({"topic": topic, "batch": batch}, f, ensure_ascii=False)
            f.write("\n")

        return None


# Securely process a batch with retry attempts
def process_lot_secure(batch_args, max_reintents=3):
    batch, topic = batch_args
    for intent in range(1, max_reintents + 1):
        try:
            return analitzar_batch(batch, topic)
        except (RateLimitError, APIError, Timeout) as e:
            print(f"⚠️ Error en lot (intent {intent}): {e}")
            time.sleep(3 * intent)  # Wait longer each retry
        except Exception as e:
            print(f"❌ Error inesperat: {e}")
            break
    return None

# Summarize opinions into a list of most common ones
def resumir_opinions(opinions_list, max_opinions=10):
    counts = Counter(opinions_list)
    ordenades = [op for op, _ in counts.most_common(max_opinions)]
    return ordenades

# Select top comments based on votes and replies
def select_comments(comentaris, sentiment, min_count=3, max_count=5):
    filtrats = [c for c in comentaris if c["sentiment"] == sentiment]
    ordenats = sorted(filtrats, key=lambda c: (c["vots"], c["respostes"]), reverse=True)
    seleccionats = ordenats[:max_count]
    if len(seleccionats) < min_count:
        restants = [c for c in filtrats if c not in seleccionats]
        random.shuffle(restants)
        seleccionats += restants[:max(0, min_count - len(seleccionats))]
    return seleccionats

# Combine multiple batch results into a final result
def combine_results(resultats):
    final = {
        "sentiments": {"positiu": 0, "negatiu": 0, "neutre": 0},
        "opinions": {"positives": [], "negatives": []},
        "comentaris": []
    }
    for r in resultats:
        if not r:
            continue
        for k in final["sentiments"]:
            final["sentiments"][k] += r["sentiments"].get(k, 0)
        final["opinions"]["positives"].extend(r["opinions"].get("positives", []))
        final["opinions"]["negatives"].extend(r["opinions"].get("negatives", []))
        final["comentaris"].extend(r["comentaris"])

    final["opinions"]["positives"] = resumir_opinions(final["opinions"]["positives"], 10)
    final["opinions"]["negatives"] = resumir_opinions(final["opinions"]["negatives"], 10)

    top_positius = select_comments(final["comentaris"], "positiu", 3, 5)
    top_negatius = select_comments(final["comentaris"], "negatiu", 3, 5)
    final["comentaris"] = top_positius + top_negatius

    return final

# Main function to analyze a CSV file
def analyze_csv(csv_name, topic):
    path = os.path.join(BASE_DIR, "data", "CSVfile", csv_name)
    output_path = os.path.join(BASE_DIR, "data", "ResultadosFiltered", csv_name.replace(".csv", "_analyzed.json"))

    # Empty the failed batches file before starting
    with open(FALLITS_FILE, "w", encoding="utf-8") as f:
        pass

    comentaris = load_comments(path)
    batches = divice_by_barches(comentaris)

    print(f"📄 Loaded {len(comentaris)} comments | {len(batches)} batches")

    args = [(b, topic) for b in batches]
    with ThreadPoolExecutor(max_workers=20) as executor:
        resultats = list(executor.map(process_lot_secure, args))

    resultat_final = combine_results(resultats)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultat_final, f, ensure_ascii=False, indent=2)

    print(f"✅ Final JSON saved to: {output_path}")
    
    # Comment if you want to keep the failed batches for reanalysis
    return resultat_final

    # Optional: Reanalyze failed batches and merge corrected results
    # Note: Currently disabled because reanalyzing is slow and often not necessary.
    '''
    if os.path.exists(FALLITS_FILE) and os.path.getsize(FALLITS_FILE) > 0:
        print("🔁 Lots fallits detectats. Reanalitzant...")
        reanalyze_failed_batches.reanalyze_batches()

        # Si tenim un fitxer amb resultats corregits, combinem-ho tot
        REANALYSIS_OUTPUT = os.path.join(BASE_DIR, "data", "ResultadosFiltered", "reanalyzed_batches.json")
        if os.path.exists(REANALYSIS_OUTPUT):
            with open(REANALYSIS_OUTPUT, "r", encoding="utf-8") as f:
                resultats_rean = json.load(f)

            # Combinar resultats originals + reanalitzats
            resultat_final = combine_results([resultat_final, resultats_rean])

            # Sobreescriure JSON final combinat
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(resultat_final, f, ensure_ascii=False, indent=2)

            print(f"✅ JSON final actualitzat amb lots reanalitzats: {output_path}")
    '''

# Main test block (disabled to use as a library)
'''
if __name__ == "__main__":
    example_csv = "reddit_praw_malta_2025-04-15.csv"
    topic = "Malta"
    analyze_csv(example_csv, topic)
'''