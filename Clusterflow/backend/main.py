# -*- coding: utf-8 -*-
"""
FastAPI Backend for Student Clustering Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Clusterflow API",
    description="ML-powered student clustering prediction API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODEL_DIR = "../model"
try:
    model = joblib.load(os.path.join(MODEL_DIR, "kprototypes_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.joblib"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "ordinal_encoder.joblib"))
    model_info = joblib.load(os.path.join(MODEL_DIR, "model_info.joblib"))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None

# Request model
class PredictionRequest(BaseModel):
    game_genre_top_1: str
    game_genre_top_2: str
    game_genre_top_3: str
    music_genre_top_1: str
    music_genre_top_2: str
    music_genre_top_3: str
    listening_hours: float

# Available genres
GAME_GENRES = ["FPS", "Sports", "Racing", "Strategy", "Puzzle", "Casual", "RPG", "MOBA"]
MUSIC_GENRES = ["Bollywood", "Classical", "Indie", "Lofi", "Pop", "Hip-Hop", "Rock", "EDM", "Devotional", "Lo-fi"]

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Clusterflow API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/available-genres")
def get_available_genres():
    """Get available game and music genres"""
    return {
        "game_genres": GAME_GENRES,
        "music_genres": MUSIC_GENRES
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict student cluster"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare categorical features
        categorical_features = [
            request.game_genre_top_1,
            request.game_genre_top_2,
            request.game_genre_top_3,
            request.music_genre_top_1,
            request.music_genre_top_2,
            request.music_genre_top_3
        ]
        
        # Encode categorical features
        categorical_encoded = encoder.transform([categorical_features])
        
        # Prepare numerical feature
        numerical_feature = np.array([[request.listening_hours]])
        numerical_scaled = scaler.transform(numerical_feature)
        
        # Combine features (both should be 2D arrays)
        features = np.concatenate([categorical_encoded, numerical_scaled], axis=1)
        
        # Predict cluster
        cluster = model.predict(features, categorical=[0, 1, 2, 3, 4, 5])[0]
        
        # Get cluster information
        cluster_profiles = model_info.get("cluster_profiles", {})
        cluster_info = cluster_profiles.get(int(cluster), {})
        
        # Extract cluster details
        cluster_name = f"Cluster {cluster}"
        description = "Student group based on gaming and music preferences"
        
        # Get top games and music from cluster profile
        top_games = []
        top_music = []
        
        if cluster_info:
            # Extract top games (first 3 categorical features)
            for i in range(3):
                game_key = f"game_genre_top_{i+1}"
                if game_key in cluster_info:
                    top_games.append(cluster_info[game_key])
            
            # Extract top music (next 3 categorical features)
            for i in range(3):
                music_key = f"music_genre_top_{i+1}"
                if music_key in cluster_info:
                    top_music.append(cluster_info[music_key])
            
            # Get average listening hours
            avg_hours = cluster_info.get("listening_hours", request.listening_hours)
        else:
            # Fallback values
            top_games = [request.game_genre_top_1, request.game_genre_top_2, request.game_genre_top_3]
            top_music = [request.music_genre_top_1, request.music_genre_top_2, request.music_genre_top_3]
            avg_hours = request.listening_hours
        
        return {
            "cluster": int(cluster),
            "cluster_name": cluster_name,
            "message": description,
            "cluster_size": len(cluster_profiles),
            "top_games": top_games[:3] if top_games else ["FPS", "Sports", "Racing"],
            "top_music": top_music[:3] if top_music else ["Pop", "Rock", "Hip-Hop"],
            "avg_listening_hours": round(float(avg_hours), 2),
            "confidence": "High"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Clusterflow API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
