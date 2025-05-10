from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importem els serveis
from app.services.findPosts import search_reddit_praw
from app.services.sortTitles import filter_titles
from app.services.analyzeSentiments import analyze_csv

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Enable CORS to allow frontend (e.g. React on localhost:5173) to call the backend API from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Use "*" to allow any origin (for development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define GET route /api/search to handle Reddit search and title filtering
@app.get("/api/search")
def search_and_filter(keyword: str):
    # Step 1: Search Reddit and save results to CSV
    csv_name = search_reddit_praw(keyword)

     # Step 2: Filter titles using a pre-trained model
    filter_titles(csv_name)

    # Step 3: Analyze sentiments using a pre-trained model
    analyze_csv(csv_name, keyword)

    return {"message": f"🔎 Cerca completada per '{keyword}'. Resultat CSV: {csv_name}. JSON result generated"}
