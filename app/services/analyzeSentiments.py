import pandas as pd
import openai
import json
import tiktoken
import time
import os
from dotenv import load_dotenv
from collections import Counter
import random

# Load environment variables from the .env file
load_dotenv()

# This line loads the API key from the .env file securely.
# It prevents hardcoding sensitive credentials directly in the script.
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Token limit per batch to avoid context overflow
MAX_TOKENS = 1800
ENCODING = tiktoken.encoding_for_model(MODEL)

# Define the root directory of the project to ensure paths are always valid
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Llegeix el fitxer CSV i extreu només els camps útils per analitzar:
#    - text del comentari
#    - puntuació (vots)
#    - nombre de respostes
#    També filtra comentaris buits per no malgastar crides a l’API.
def carregar_comentaris(csv_path):
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


# Divideix la llista de comentaris en lots més petits segons el nombre total de tokens.
# Això assegura que cada petició a l’API estigui dins del límit de context.
def dividir_per_batches(comentaris):
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


# Envia un lot de comentaris a l’API de ChatGPT amb un prompt estructurat.
#    - Classifica sentiments
#    - Resumeix opinions positives i negatives
#    - Retorna comentaris destacats
# El model retorna un JSON que conté: sentiments totals, opinions resumides i comentaris destacats.
def analitzar_batch(batch, topic):
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
        resposta = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return json.loads(resposta.choices[0].message.content)
    except Exception as e:
        print("⚠️ Error in batch:", e)
        return None


# Rep una llista d’opinions i retorna les més repetides.
# Serveix per mostrar només les 10 més representatives.
def resumir_opinions(opinions_list, max_opinions=10):
    counts = Counter(opinions_list)
    ordenades = [op for op, _ in counts.most_common(max_opinions)]
    return ordenades


#Filtra els comentaris per sentiment i retorna els més rellevants segons:
#    - Vots (upvotes)
#    - Nombre de respostes
#   Torna entre 3 i 5 comentaris per tipus.
def seleccionar_comentaris(comentaris, sentiment, min_count=3, max_count=5):
    filtrats = [c for c in comentaris if c["sentiment"] == sentiment]
    ordenats = sorted(filtrats, key=lambda c: (c["vots"], c["respostes"]), reverse=True)
    seleccionats = ordenats[:max_count]
    if len(seleccionats) < min_count:
        restants = [c for c in filtrats if c not in seleccionats]
        random.shuffle(restants)
        seleccionats += restants[:max(0, min_count - len(seleccionats))]
    return seleccionats


#Combina tots els resultats dels batches en un sol JSON:
#    - Suma els sentiments
#    - Fusiona les opinions
#    - Selecciona els comentaris més destacats per sentiment
def combinar_resultats(resultats):
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

    # Resumir i limitar opinions
    final["opinions"]["positives"] = resumir_opinions(final["opinions"]["positives"], 10)
    final["opinions"]["negatives"] = resumir_opinions(final["opinions"]["negatives"], 10)

    # Seleccionar 3-5 comentaris destacats per sentiment
    top_positius = seleccionar_comentaris(final["comentaris"], "positiu", 3, 5)
    top_negatius = seleccionar_comentaris(final["comentaris"], "negatiu", 3, 5)
    final["comentaris"] = top_positius + top_negatius

    return final


# Funció principal que:
#    - Carrega el CSV
#    - Divideix en batches
#    - Envia cada lot a ChatGPT
#    - Combina els resultats finals
#    - Desa el JSON al disc
def analitzar_csv(csv_name, topic):
    path = os.path.join(BASE_DIR, "data", "CSVfile", csv_name)
    output_path = os.path.join(BASE_DIR, "data", "ResultadosFiltered", csv_name.replace(".csv", "_analyzed.json"))

    comentaris = carregar_comentaris(path)
    batches = dividir_per_batches(comentaris)

    print(f"📄 Loaded {len(comentaris)} comments | {len(batches)} batches")

    resultats = []
    for i, b in enumerate(batches):
        print(f"🔍 Analyzing batch {i+1}/{len(batches)}...")
        resultat = analitzar_batch(b, topic)
        resultats.append(resultat)
        time.sleep(1.5) # Pausa per evitar superar límits de l'API

    resultat_final = combinar_resultats(resultats)

    # Desa el resultat final al disc
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultat_final, f, ensure_ascii=False, indent=2)

    print(f"✅ Final JSON saved to: {output_path}")
    return resultat_final

if __name__ == "__main__":
    example_csv = "reddit_praw_malta_2025-04-15.csv"
    topic = "Malta"
    analitzar_csv(example_csv, topic)