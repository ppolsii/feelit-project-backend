from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importem els serveis
from app.services.findPosts import search_reddit_praw
from app.services.sortTitles import filter_titles
from app.services.analyzeSentiments import analyze_csv

from fastapi.responses import JSONResponse
import os
import json
import urllib.parse

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

# Define GET route /api/search to handle Reddit search
@app.get("/api/search")
def search_and_filter(keyword: str): #Keyword is got from the frontend
    # Step 1: Search Reddit and save results to CSV
    csv_name = search_reddit_praw(keyword)

     # Step 2: Filter titles using a pre-trained model
    filter_titles(csv_name)

    # Step 3: Analyze sentiments using a pre-trained model
    analyze_csv(csv_name, keyword)

    return {"message": f"Searcj completed for '{keyword}'. CSV results: {csv_name}. JSON result generated"}


# Define GET route /search/{keyword} to retrieve search results
@app.get("/search/{keyword}")
def get_search_results(keyword: str):
    # Decode URL encoding and replace spaces
    decoded_keyword = urllib.parse.unquote(keyword)
    sanitized_keyword = decoded_keyword.replace(" ", "_")

    # Path to where the results are stored
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ResultadosFiltered"))

    # Check if the .json file for the given keyword exists
    matching_files = [
        f for f in os.listdir(base_path)
        if f.startswith(f"reddit_praw_{sanitized_keyword}") and f.endswith("_analyzed.json")
    ]

    # If no matching files are found, return a 404 error
    if not matching_files:
        return JSONResponse(status_code=404, content={"error": f"No results found for '{keyword}'."})

    # Selected the most recent file, in casa of multiple matches
    matching_files.sort(reverse=True)
    filepath = os.path.join(base_path, matching_files[0])

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load results: {str(e)}"})
    