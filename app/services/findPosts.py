import csv
import praw
import os
import datetime as dt
from pathlib import Path

# ==== 1️⃣ Definim la funció que inicialitza Reddit i fa la cerca ====

def search_reddit_praw(keyword):
    # 🔐 Carreguem variables dins la funció (ja tenim .env carregat des de main.py)
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')

    # ❗ Verificació opcional per depuració
    if not client_id or not client_secret or not user_agent:
        raise RuntimeError("❌ Alguna variable d'entorn (REDDIT_CLIENT_ID, ...) no s'ha carregat!")

    # Inicialitzar Reddit API
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    # ✅ Ruta base del projecte
    main_dir = Path(__file__).resolve().parents[2]  # Pujar 2 nivells

    # Carpeta per guardar els .csv
    folder = os.path.join(main_dir, 'data', 'CSVfile')

    # Nom del fitxer de sortida
    sanitized_keyword = keyword.replace(" ", "_")
    date = dt.date.today()
    csv_filename = os.path.join(folder, f'reddit_praw_{sanitized_keyword}_{date}.csv')

    os.makedirs(folder, exist_ok=True)

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            '#',
            'Post Title',
            'Post Score (Upvotes)',
            'Post Number of Comments',
            'Comment Text',
            'Comment Score (Upvotes)',
            'Number of Replies'
        ])

        print(f"\n🔍 Buscant posts sobre: {keyword} (PRAW)\n")
        subreddit = reddit.subreddit('all')

        num = 1
        for submission in subreddit.search(keyword, sort='relevance', limit=100):
            if keyword.lower() not in submission.title.lower():
                continue
            if submission.num_comments == 0:
                continue

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                comment_text = comment.body.strip().replace('\n', ' ')
                comment_score = comment.score
                num_replies = len(comment.replies)

                if len(comment_text) > 30 and comment_score >= 1:
                    csv_writer.writerow([
                        num,
                        submission.title,
                        submission.score,
                        submission.num_comments,
                        comment_text,
                        comment_score,
                        num_replies
                    ])
                    num += 1

    print(f"\n✅ Resultats guardats en '{csv_filename}'")
    return csv_filename
