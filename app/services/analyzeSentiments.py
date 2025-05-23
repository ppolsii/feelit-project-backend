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
from openai import OpenAIError
import re
from openai import OpenAI
# import reanalyze_failed_batches

# Load environment variables from the .env file
load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
# print("🔑 Clau API carregada:", openai.api_key is not None)

MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = 1800
ENCODING = tiktoken.encoding_for_model(MODEL)

# Define the base directory for file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Create a directory for failed batches if it doesn't exist
# Note: Its implemented but not used, due to the time it takes to reanalyze the batches --> Not worth it.
FAILED_PATH = os.path.join(BASE_DIR, "data", "LotesFallits")
os.makedirs(FAILED_PATH, exist_ok=True)
FALLITS_FILE = os.path.join(FAILED_PATH, "batches_fallits.jsonl")

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


# ==== Main function to analyze a single batch of comments using OpenAI API ====
def analitzar_batch(batch, topic):
    # Build the prompt with the topic and comments
    texts = [c["text"] for c in batch]
    prompt = f"""
Analyze the following Reddit comments about the topic \"{topic}\":

1. Classify each comment as 'positiu', 'negatiu', or 'neutre'.
2. Provide standalone, self-contained summaries of positive and negative opinions (avoid vague replies, partial phrases, or references to other comments).
3. Include the most relevant comments with high votes or replies.
4. Translate opinions and comentaris to English.

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
        client = openai.OpenAI()  # Create an OpenAI client instance using the API key

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Fix broken escape sequences
        content = re.sub(r'\\(?![nrt"\\/bfu])', r'\\\\', content)

        # Attempt to parse the response as JSON
        try:
            return json.loads(content)

        # Handle invalid JSON responses
        except json.JSONDecodeError as e:
            print("JSON recived is not valid (first lines):")
            print(content[:1000])  # Limit output to 1000 chars for debugging
            print("Error:", e)

            # If response looks incomplete or truncated, check for unclosed braces
            if not content.strip().endswith("}") or content.count("{") > content.count("}"):
                print("Answer probably truncated. Batch saved to file.")

            # Save failed batch for later reanalysis
            with open(FALLITS_FILE, "a", encoding="utf-8") as f:
                json.dump({"topic": topic, "batch": batch}, f, ensure_ascii=False)
                f.write("\n")

            return None

    except Exception as e:
        # Handle general errors
        print("Important error on batch:", e)

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
        except OpenAIError as e:
            print(f"Faile on batch (try {intent}): {e}")
            time.sleep(3 * intent)  # Wait longer each retry
        except Exception as e:
            print(f"Unexpected error on batch: {e}")
            break
    return None

# Summarize opinions into a list of most common ones
def summarize_opinions(opinions_list, max_opinions=10):
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
    # Final result structure
    final = {
        "sentiments": {"positiu": 0, "negatiu": 0, "neutre": 0},
        "opinions": {"positives": [], "negatives": []},
        "comentaris": []
    }

    # Iterate for each result of process_lot_secure()
    for r in resultats:
        if not r: # Skip if the batch failed
            continue
        for k in final["sentiments"]:
            # Add the sentiments from the batch to the final result
            final["sentiments"][k] += r["sentiments"].get(k, 0)
        
        # Add the opinions from the batch to the final result
        final["opinions"]["positives"].extend(r["opinions"].get("positives", []))
        final["opinions"]["negatives"].extend(r["opinions"].get("negatives", []))

        # Add the comments from the batch to the final result
        final["comentaris"].extend(r["comentaris"])

    # Summarize opinions and select top comments
    final["opinions"]["positives"] = summarize_opinions(final["opinions"]["positives"], 10)
    final["opinions"]["negatives"] = summarize_opinions(final["opinions"]["negatives"], 10)

    # Select top comments for positive and negative sentiments
    top_positius = select_comments(final["comentaris"], "positiu", 3, 5)
    top_negatius = select_comments(final["comentaris"], "negatiu", 3, 5)

    # Remove the selected comments from the main list
    final["comentaris"] = top_positius + top_negatius

    return final

# Main function to analyze a CSV file
def analyze_csv(csv_name, topic):
    # Define the path to the CSV file
    path = os.path.join(BASE_DIR, "data", "CSVfile", os.path.basename(csv_name))

    filename_only = os.path.basename(csv_name)
    output_path = os.path.join(BASE_DIR, "data", "ResultadosFiltered", filename_only.replace(".csv", "_analyzed.json"))

    # Empty the failed batches file before starting
    with open(FALLITS_FILE, "w", encoding="utf-8") as f:
        pass

    # Load comments from the CSV file
    comentaris = load_comments(path)

    # Check if there are comments to analyze
    batches = divice_by_barches(comentaris)

    print(f"Loaded {len(comentaris)} comments | {len(batches)} batches")

    # Parallelize the analysis of batches
    args = [(b, topic) for b in batches]

    # Use ThreadPoolExecutor to process batches in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        resultats = list(executor.map(process_lot_secure, args))

    # Combine results from all batches
    resultat_final = combine_results(resultats)

    # Save the final result to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultat_final, f, ensure_ascii=False, indent=2)

    print(f"Final JSON saved to: {output_path}")
    
    # Comment if you want to keep the failed batches for reanalysis
    return resultat_final

    # Optional: Reanalyze failed batches and merge corrected results
    # Note: Currently disabled because reanalyzing is slow and often not necessary.
    '''
    if os.path.exists(FALLITS_FILE) and os.path.getsize(FALLITS_FILE) > 0:
        print("Lots fallits detectats. Reanalitzant...")
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

            print(f"JSON final actualitzat amb lots reanalitzats: {output_path}")
    '''

# Main test block (disabled to use as a library)
'''
if __name__ == "__main__":
    example_csv = "reddit_praw_Trip_to_Malta_2025-05-22.csv"
    topic = "Malta"
    analyze_csv(example_csv, topic)
'''
