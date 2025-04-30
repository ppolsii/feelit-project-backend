import csv
import praw
import os
import datetime as dt
from pathlib import Path

# ==== <-- 1 --> Main function to search Reddit and save data to a CSV ====
def search_reddit_praw(keyword):
    # Load Reddit API credentials from .env (already loaded in main.py)
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')

    #  Debug check: ensure all credentials are present
    if not client_id or not client_secret or not user_agent:
        raise RuntimeError("❌ Missing environment variables: REDDIT_CLIENT_ID, etc... - not found. Please check your environment!")

    # Initialize Reddit API client
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    # Define project root path (2 levels up from this file)
    main_dir = Path(__file__).resolve().parents[2]  # Two levels up from this file

    # Folder to store generated CSVs (.csv)
    folder = os.path.join(main_dir, 'data', 'CSVfile')

    # Create sanitized CSV filename using keyword and current date
    sanitized_keyword = keyword.replace(" ", "_")
    date = dt.date.today()
    csv_filename = os.path.join(folder, f'reddit_praw_{sanitized_keyword}_{date}.csv')

    os.makedirs(folder, exist_ok=True)

    # Open CSV file for writing
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

        print(f"\n🔍 Searching Reddit for: {keyword} (PRAW)\n")
        subreddit = reddit.subreddit('all')

        num = 1
        # Search up to 100 Reddit posts matching the keyword, sorted by relevance
        for submission in subreddit.search(keyword, sort='relevance', limit=100):
            # Only include posts that actually contain the keyword and have comments
            if keyword.lower() not in submission.title.lower():
                continue
            if submission.num_comments == 0:
                continue

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                comment_text = comment.body.strip().replace('\n', ' ')
                comment_score = comment.score
                num_replies = len(comment.replies)

                # Save comment if it's meaningful (long enough and has upvotes)
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

    print(f"\n✅ Results saved in '{csv_filename}'")
    return csv_filename
