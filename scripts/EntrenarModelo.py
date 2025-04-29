# EntrenarModelo.py

# pip install transformers datasets scikit-learn
# pip install "transformers[torch]"
# pip install "accelerate>=0.26.0"

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# === 1️⃣ Carregar CSV etiquetat ===
csv_path = Path("./data/titles_training_dataset.csv").resolve()
df = pd.read_csv(csv_path, sep='|')
df = df.dropna(subset=['Label'])  # Només títols que tinguin etiqueta
df['Label'] = df['Label'].astype(int)

# === 2️⃣ Dividir en training i test ===
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Post Title'].tolist(), df['Label'].tolist(), test_size=0.2, random_state=42
)

# === 3️⃣ Tokenització ===
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# === 4️⃣ Carregar model ===
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# === 5️⃣ Configuració d'entrenament ===
training_args = TrainingArguments(
    output_dir='./app/models/model_output',  # Guardar model a lloc professional
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',  # Carpeta de logs
    logging_steps=10,
)

# === 6️⃣ Entrenar ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# === 7️⃣ Guardar el model manualment ===
trainer.save_model('./app/models/model_output')
print("\n✅ Model entrenat i guardat a './app/models/model_output'")
