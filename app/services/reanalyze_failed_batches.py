import json
import os
from analyzeSentiments import analitzar_batch, combine_results, BASE_DIR

# Path to the file containing failed batches
FALLITS_FILE = os.path.join(BASE_DIR, "data", "LotesFallits", "batches_fallits.jsonl")
# Path where corrected results will be saved
REANALYSIS_OUTPUT = os.path.join(BASE_DIR, "data", "ResultadosFiltered", "reanalyzed_batches.json")

# Load all previously saved failed batches
def load_failed_batches():
    if not os.path.exists(FALLITS_FILE):
        print("⚠️ No s’ha trobat cap fitxer de lots fallits.")
        return []

    with open(FALLITS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lots = []
    for line in lines:
        try:
            lots.append(json.loads(line))
        except Exception as e:
            print("Error carregant una línia:", e)
    return lots

# Reanalyze failed batches and filter only those that succeed
def reanalyze_batches():
    lots = load_failed_batches()
    if not lots:
        print("No hi ha lots pendents.")
        return

    resultats_bons = []
    lots_restants = []

    print(f"Reanalitzant {len(lots)} lots fallits...")

    for i, lot_info in enumerate(lots):
        topic = lot_info["topic"]
        batch = lot_info["batch"]

        print(f"🔍 Lot {i+1}/{len(lots)}...")
        resultat = analitzar_batch(batch, topic)

        if resultat:
            resultats_bons.append(resultat)
        else:
            # If it still fails, keep it for another attempt
            lots_restants.append(lot_info)

    # Save successfully reanalyzed batches combined into one file
    if resultats_bons:
        resultat_final = combine_results(resultats_bons)
        with open(REANALYSIS_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(resultat_final, f, ensure_ascii=False, indent=2)
        print(f"✅ Resultats reanalitzats guardats a: {REANALYSIS_OUTPUT}")

    # Update the file keeping only batches that still fail
    with open(FALLITS_FILE, "w", encoding="utf-8") as f:
        for lot in lots_restants:
            json.dump(lot, f, ensure_ascii=False)
            f.write("\n")

    if lots_restants:
        print(f"⚠️ {len(lots_restants)} lots encara fallen i s’han tornat a guardar.")
    else:
        print("Tots els lots s’han reanalitzat correctament!")

if __name__ == "__main__":
    reanalyze_batches()
      # If you want to execute directly, uncomment the line above