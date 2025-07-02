from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
from pathlib import Path

# ==== Main function to classify and filter titles from a CSV file====
# Filters a CSV file by classifying post titles using DistilBERT 
# and keeps only the ones predicted as relevant (label 1).
# Overwrites the original CSV with the filtered results.
def filter_titles(csv_path: str):

    # Define directory where this script is located
    this_dir = Path(__file__).resolve().parent

    # Full path to the saved model directory
    model_path = (this_dir.parent / "models" / "model_output").resolve()

    # Load pre-trained model and tokenizer from local directory
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_path, local_files_only=True
    )

    # Set the model to evaluation mode (for prediction only)
    # This disables things like dropout and ensures consistent results
    model.eval() 
    
    input_csv = Path(csv_path).resolve()
    print("sortTitles_service.py → input_csv:", input_csv)

    # Read the original CSV
    df_full = pd.read_csv(input_csv)

    # Remove duplicate titles before classification
    unique_titles = df_full[['Post Title']].drop_duplicates()

    # Apply the model to each title and store predictions
    predictions = []
    for title in unique_titles['Post Title']:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)

    # Add predictions as a new column
    unique_titles['Predicted Label'] = predictions

    # Keep only titles classified as relevant (label == 1)
    interesting_titles = unique_titles[unique_titles['Predicted Label'] == 1]['Post Title']
    df_filtered = df_full[df_full['Post Title'].isin(interesting_titles)]

    # Overwrite the original CSV with filtered results
    df_filtered.to_csv(input_csv, index=False)
    print(f"\nOverwritten CSV with {len(df_filtered)} entries: {input_csv}")
