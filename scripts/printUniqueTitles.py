import os
import pandas as pd

folder_path = './data/CSVfile'


for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"\n📄 Archivo: {filename}")

        # Leer el CSV
        df = pd.read_csv(file_path)

        # Obtener títulos únicos eliminando duplicados
        unique_titles = df['Post Title'].drop_duplicates()

        # Imprimir títulos únicos
        print(f"Títulos únicos en '{filename}':")
        for title in unique_titles:
            print(f" - {title}")
