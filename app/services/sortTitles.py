from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
from pathlib import Path
import os

# 1) Obtener la ruta absoluta de la carpeta donde está este script
this_dir = Path(__file__).resolve().parent

# 2) Construir la ruta absoluta del modelo
model_path = (this_dir.parent / "models" / "model_output").resolve()

# ✅ Carga del modelo y tokenizer desde carpeta local
model = DistilBertForSequenceClassification.from_pretrained(
    model_path, local_files_only=True
)
tokenizer = DistilBertTokenizerFast.from_pretrained(
    'distilbert-base-uncased', local_files_only=True
)
model.eval()

# 🔥 Funció que reemplaça sys.argv
def filter_titles(csv_path: str):
    """
    Filtra un CSV donat, aplicant el model DistilBERT per seleccionar només els títols rellevants.
    Sobreescriu el CSV original amb els títols filtrats.
    """
    input_csv = Path(csv_path).resolve()
    print("sortTitles_service.py → input_csv:", input_csv)

    # Leer CSV
    df_full = pd.read_csv(input_csv)

    # Eliminar duplicats només per a classificació
    unique_titles = df_full[['Post Title']].drop_duplicates()

    # Aplicar model als títols
    predictions = []
    for title in unique_titles['Post Title']:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)

    # Afegir columna de predicció
    unique_titles['Predicted Label'] = predictions

    # Filtrar títols interessants (label = 1)
    interesting_titles = unique_titles[unique_titles['Predicted Label'] == 1]['Post Title']
    df_filtered = df_full[df_full['Post Title'].isin(interesting_titles)]

    # Sobreescriure el CSV original amb el resultat final
    df_filtered.to_csv(input_csv, index=False)
    print(f"\n✅ S'ha sobrescrit el CSV amb {len(df_filtered)} entrades: {input_csv}")
