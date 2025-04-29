import csv
import praw
import os
import datetime as dt
from dotenv import load_dotenv

# ==== 1️⃣ Cargar variables de entorno ====

# Calcular la ruta al .env (dos carpetas arriba de este script)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=env_path)

# Leer las variables del .env
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_agent = os.getenv('USER_AGENT')


# ==== 2️⃣ Inicializar Reddit API ====
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

def search_reddit_praw(keyword):

    # Ruta del directorio principal (sube un nivel desde main4.py)
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Carpeta para guardar CSVs
    # FOR TRAINING: folder = os.path.join(main_dir, 'CSVsTraining')
    folder = os.path.join(main_dir, 'data', 'CSVfile')
    
    # Construir filename reemplazando espacios por _
    sanitized_keyword = keyword.replace(" ", "_")
    date = dt.date.today()
    csv_filename = os.path.join(folder, f'reddit_praw_{sanitized_keyword}_{date}.csv')
    
    # Crear carpeta si no existe
    os.makedirs(folder, exist_ok=True)


    #csv_filename = f'CSVs/reddit_praw_{keyword.replace(" ", "_")}.csv'

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header
        csv_writer.writerow([
            '#', 
            'Post Title', 
            #'Post URL', 
            'Post Score (Upvotes)',
            'Post Number of Comments', 
            'Comment Text', 
            #'Username',
            'Comment Score (Upvotes)', 
            'Number of Replies'
        ])

        print(f"\n🔍 Buscando posts sobre: {keyword} (PRAW)\n")
        subreddit = reddit.subreddit('all')

        num = 1

        # Buscamos hasta 100 posts relevantes
        for submission in subreddit.search(keyword, sort='relevance', limit=100):
            # Aseguramos que el título contenga la palabra clave
            if keyword.lower() not in submission.title.lower():
                continue

            if submission.num_comments == 0:
                continue  # Ignora posts sin comentarios

            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list():
                # username = comment.author.name if comment.author else "[deleted]"
                comment_text = comment.body.strip().replace('\n', ' ')
                comment_score = comment.score
                num_replies = len(comment.replies)

                # Solo comentarios relevantes
                if len(comment_text) > 30 and comment_score >= 1:
                    csv_writer.writerow([
                        num,
                        submission.title,
                        # submission.url,
                        submission.score,
                        submission.num_comments,
                        comment_text,
                        # username,
                        comment_score,
                        num_replies
                    ])
                    num += 1

    print(f"\n✅ Resultados guardados en '{csv_filename}'")
    return csv_filename

# ==== Ejecutar búsqueda ====

#if __name__ == "__main__":
    #keyword = input("\n🔎 Ingrese el tema a buscar en Reddit: ")
    #search_reddit_praw(keyword)
