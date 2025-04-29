from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importem els serveis
from app.services.findPosts import search_reddit_praw
from app.services.sortTitles import filter_titles

# ✅ Creem l'aplicació FastAPI
app = FastAPI()

# ✅ Configuració de CORS per permetre connexions des de localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # o "*" si vols permetre-ho tot en desenvolupament
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Definim la ruta /api/search
@app.get("/api/search")
def search_and_filter(keyword: str):
    csv_name = search_reddit_praw(keyword)
    filter_titles(csv_name)
    return {"message": f"🔎 Cerca completada per '{keyword}'. Resultat: {csv_name}"}
