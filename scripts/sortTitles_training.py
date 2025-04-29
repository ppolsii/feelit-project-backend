# sortTitles_training.py
import os
import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

print("sortTitles.py → os.getcwd():", os.getcwd())
print("sortTitles.py → __file__:", __file__)

# 1) Ruta absoluta de la carpeta donde ESTÁ este script (sortTitles_training.py)
this_dir = Path(__file__).resolve().parent

# 2) Ruta absoluta del modelo (aquí asumo que 'model_output' está en la MISMA carpeta 'main')
#    Si tu 'model_output' estuviera en la carpeta padre, reemplaza con: this_dir.parent / "model_output"
model_path = (this_dir.parent / "models" / "model_output").resolve()

# 3) Convertir la ruta a POSIX para que huggingface no la interprete como repo remoto
model_path_str = model_path.as_posix()
print("sortTitles.py → model_path:", model_path_str)

# 4) Cargar modelo indicando 'local_files_only=True' (evita buscar en Hugging Face Hub)
model = DistilBertForSequenceClassification.from_pretrained(model_path_str, local_files_only=True)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model.eval()

# 5) CSV que le pases como argumento
if len(sys.argv) < 2:
    print("❌ Error: Debes pasar el nombre del archivo CSV como argumento.")
    sys.exit(1)

input_csv = Path(sys.argv[1]).resolve().as_posix()
print("sortTitles.py → input_csv:", input_csv)

# 6) Leer y filtrar
df_full = pd.read_csv(input_csv)
unique_titles = df_full[['Post Title']].drop_duplicates()

predictions = []
for title in unique_titles['Post Title']:
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    predictions.append(pred)

unique_titles['Predicted Label'] = predictions
interesting_titles = unique_titles[unique_titles['Predicted Label'] == 1]['Post Title']
df_filtered = df_full[df_full['Post Title'].isin(interesting_titles)]



# === 5️⃣ Guardar resultado final ===
output_folder = 'data/ResultadosFiltered'
os.makedirs(output_folder, exist_ok=True)
filename = os.path.basename(input_csv).replace('.csv', '_filtered112121dasdad21.csv')
output_csv = os.path.join(output_folder, filename)
df_filtered.to_csv(output_csv, index=False)

print(f"\n✅ Archivo final guardado con {len(df_filtered)} entradas: {output_csv}")
