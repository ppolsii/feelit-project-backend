from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
import os

# === 1️⃣ Cargar modelo entrenado ===
model_path = './model_output'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model.eval()

# === 2️⃣ Leer CSV de nuevos títulos ===
input_csv = 'CSVsTraining/reddit_praw_nintendo_switch.csv' # Ajusta aquí el nombre de tu CSV
df = pd.read_csv(input_csv)

# Si hay duplicados, los quitamos
df = df.drop_duplicates(subset=['Post Title'])

# === 3️⃣ Aplicar modelo a cada título ===
predictions = []

for title in df['Post Title']:
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        predictions.append(pred)

# === 4️⃣ Añadir columna con predicciones ===
df['Predicted Label'] = predictions

# === 5️⃣ Guardar resultado en nuevo CSV ===
output_folder = 'data/ResultadosLabeled'  # Nombre de la carpeta donde guardar
os.makedirs(output_folder, exist_ok=True)  # Crea la carpeta si no existe

# Extraemos solo el nombre del archivo, sin la ruta previa
filename = os.path.basename(input_csv).replace('.csv', '_labeled.csv')

# Construimos la nueva ruta
output_csv = os.path.join(output_folder, filename)
df.to_csv(output_csv, index=False)

print(f"\n✅ Predicciones completadas. Archivo guardado como: {output_csv}")
