# -*- coding: utf-8 -*-
"""
Streamlit Frontend for Student Clustering Prediction
"""
import streamlit as st
import requests
import json
from typing import Dict, Any

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Clusterflow",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force title update with JavaScript
st.markdown("""
    <script>
        document.title = "Clusterflow";
    </script>
""", unsafe_allow_html=True)

# Custom CSS - Grey Liquid Glass Theme
st.markdown("""
    <style>
    /* Grey animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #1a1a1a, #2d2d2d, #3a3a3a, #2d2d2d, #1a1a1a);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main content area - grey glass effect */
    .main {
        padding: 2rem;
        background: rgba(75, 85, 99, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(209, 213, 219, 0.2);
        box-shadow: 0 8px 32px 0 rgba(107, 114, 128, 0.3);
    }
    
    /* Sidebar grey glass effect */
    [data-testid="stSidebar"] {
        background: rgba(55, 65, 81, 0.2);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(209, 213, 219, 0.2);
    }
    
    /* Grey glass cards for inputs */
    .stSelectbox, .stSlider {
        background: rgba(75, 85, 99, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(209, 213, 219, 0.2);
    }
    
    /* Stylish title with grey glow */
    h1 {
        animation: fadeInGlow 1.5s ease-in;
        color: #ffffff !important;
        text-shadow: 0 0 30px rgba(156, 163, 175, 0.8),
                     0 0 60px rgba(107, 114, 128, 0.6),
                     0 0 90px rgba(75, 85, 99, 0.4);
        font-weight: 900;
        font-size: 3.5rem;
    }
    
    h3 {
        color: rgba(255, 255, 255, 0.95) !important;
        text-shadow: 0 2px 15px rgba(156, 163, 175, 0.4);
    }
    
    @keyframes fadeInGlow {
        from { 
            opacity: 0; 
            transform: translateY(-30px);
            text-shadow: 0 0 0px rgba(156, 163, 175, 0);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
            text-shadow: 0 0 30px rgba(156, 163, 175, 0.8),
                         0 0 60px rgba(107, 114, 128, 0.6),
                         0 0 90px rgba(75, 85, 99, 0.4);
        }
    }
    
    /* Grey glass button with glow */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, rgba(75, 85, 99, 0.3), rgba(107, 114, 128, 0.3));
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        color: #ffffff;
        height: 3em;
        border-radius: 15px;
        font-size: 18px;
        font-weight: bold;
        border: 1px solid rgba(209, 213, 219, 0.3);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(107, 114, 128, 0.4),
                    0 0 30px rgba(75, 85, 99, 0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(75, 85, 99, 0.5), rgba(107, 114, 128, 0.5));
        transform: translateY(-3px);
        box-shadow: 0 8px 35px rgba(107, 114, 128, 0.6),
                    0 0 50px rgba(156, 163, 175, 0.4);
        border: 1px solid rgba(209, 213, 219, 0.5);
    }
    
    /* Grey glass info boxes */
    .stAlert {
        background: rgba(75, 85, 99, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(209, 213, 219, 0.2);
        color: rgba(255, 255, 255, 0.95);
    }
    
    /* Metric cards - grey glass effect */
    [data-testid="stMetricValue"] {
        color: #d1d5db;
        font-weight: bold;
        text-shadow: 0 0 15px rgba(209, 213, 219, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* Text color adjustments */
    .stMarkdown, p, label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Subheaders with grey glow */
    h2, h4 {
        color: #e5e7eb !important;
        text-shadow: 0 0 20px rgba(229, 231, 235, 0.4);
    }
    
    /* Download button with grey accent */
    .stDownloadButton>button {
        background: rgba(107, 114, 128, 0.2);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(156, 163, 175, 0.4);
        color: #d1d5db;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(107, 114, 128, 0.3);
    }
    
    .stDownloadButton>button:hover {
        background: rgba(107, 114, 128, 0.35);
        box-shadow: 0 8px 35px rgba(107, 114, 128, 0.5);
        border: 1px solid rgba(156, 163, 175, 0.6);
        color: #ffffff;
    }
    
    /* Success messages with green glass */
    .stSuccess {
        background: rgba(16, 185, 129, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 15px;
        color: #6ee7b7;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }
    
    /* Error messages with red glass */
    .stError {
        background: rgba(239, 68, 68, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 15px;
        color: #fca5a5;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    }
    
    /* Info messages with grey glass */
    .stInfo {
        background: rgba(75, 85, 99, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(156, 163, 175, 0.4);
        border-radius: 15px;
        color: #e5e7eb;
        box-shadow: 0 0 25px rgba(75, 85, 99, 0.2);
    }
    
    /* Input fields grey glass effect */
    input, select, textarea {
        background: rgba(75, 85, 99, 0.1) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(209, 213, 219, 0.2) !important;
        border-radius: 12px !important;
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border: 1px solid rgba(156, 163, 175, 0.6) !important;
        box-shadow: 0 0 20px rgba(107, 114, 128, 0.3) !important;
    }
    
    /* Slider styling with grey accent */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, rgba(75, 85, 99, 0.5), rgba(107, 114, 128, 0.5));
    }
    
    /* Selectbox dropdown grey theme */
    [data-baseweb="select"] {
        background: rgba(75, 85, 99, 0.2);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
    }
    
    /* Divider lines with grey glow */
    hr {
        border-color: rgba(156, 163, 175, 0.3);
        box-shadow: 0 0 15px rgba(107, 114, 128, 0.2);
    }
    
    /* Column containers with glass effect */
    [data-testid="column"] {
        background: rgba(75, 85, 99, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(209, 213, 219, 0.15);
        box-shadow: 0 4px 20px rgba(107, 114, 128, 0.2);
    }
    
    /* Warning messages */
    .stWarning {
        background: rgba(245, 158, 11, 0.15);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(251, 191, 36, 0.4);
        border-radius: 15px;
        color: #fcd34d;
        box-shadow: 0 0 25px rgba(251, 191, 36, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Load ML models directly
import joblib
import numpy as np
import os

@st.cache_resource
def load_models():
    """Load ML models"""
    try:
        model_dir = "../model"
        model = joblib.load(os.path.join(model_dir, "kprototypes_model.joblib"))
        scaler = joblib.load(os.path.join(model_dir, "standard_scaler.joblib"))
        encoder = joblib.load(os.path.join(model_dir, "ordinal_encoder.joblib"))
        model_info = joblib.load(os.path.join(model_dir, "model_info.joblib"))
        return model, scaler, encoder, model_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Load models
model, scaler, encoder, model_info = load_models()

def check_api_health() -> bool:
    """Check if models are loaded"""
    return model is not None

def get_available_genres() -> Dict[str, list]:
    """Get available genres"""
    return {
        "game_genres": ["FPS", "Sports", "Racing", "Strategy", "Puzzle", "Casual", "RPG", "MOBA"],
        "music_genres": ["Bollywood", "Classical", "Indie", "Lofi", "Pop", "Hip-Hop", "Rock", "EDM", "Devotional", "Lo-fi"]
    }

def make_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction using loaded models"""
    if model is None:
        return {"success": False, "error": "Models not loaded"}
    
    try:
        # Prepare categorical features
        categorical_features = [
            data["game_genre_top_1"],
            data["game_genre_top_2"],
            data["game_genre_top_3"],
            data["music_genre_top_1"],
            data["music_genre_top_2"],
            data["music_genre_top_3"]
        ]
        
        # Encode categorical features
        categorical_encoded = encoder.transform([categorical_features])
        
        # Prepare numerical feature
        numerical_feature = np.array([[data["listening_hours"]]])
        numerical_scaled = scaler.transform(numerical_feature)
        
        # Combine features
        features = np.concatenate([categorical_encoded, numerical_scaled], axis=1)
        
        # Predict cluster
        cluster = model.predict(features, categorical=[0, 1, 2, 3, 4, 5])[0]
        
        # Get cluster information
        cluster_profiles = model_info.get("cluster_profiles", {})
        cluster_info = cluster_profiles.get(int(cluster), {})
        
        # Extract cluster details
        cluster_name = f"Cluster {cluster}"
        description = "Student group based on gaming and music preferences"
        
        # Get top games and music
        top_games = []
        top_music = []
        
        if cluster_info:
            for i in range(3):
                game_key = f"game_genre_top_{i+1}"
                if game_key in cluster_info:
                    top_games.append(cluster_info[game_key])
            
            for i in range(3):
                music_key = f"music_genre_top_{i+1}"
                if music_key in cluster_info:
                    top_music.append(cluster_info[music_key])
            
            avg_hours = cluster_info.get("listening_hours", data["listening_hours"])
        else:
            top_games = [data["game_genre_top_1"], data["game_genre_top_2"], data["game_genre_top_3"]]
            top_music = [data["music_genre_top_1"], data["music_genre_top_2"], data["music_genre_top_3"]]
            avg_hours = data["listening_hours"]
        
        result = {
            "cluster": int(cluster),
            "cluster_name": cluster_name,
            "message": description,
            "cluster_size": len(cluster_profiles),
            "top_games": top_games[:3] if top_games else ["FPS", "Sports", "Racing"],
            "top_music": top_music[:3] if top_music else ["Pop", "Rock", "Hip-Hop"],
            "avg_listening_hours": round(float(avg_hours), 2),
            "confidence": "High"
        }
        
        return {"success": True, "data": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Main App
def main():
    # Header with stylish design
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='
                font-family: "Arial Black", "Helvetica Neue", Arial, sans-serif;
                font-size: 3.5rem;
                font-weight: 900;
                color: #6B46C1;
                margin-bottom: 0.5rem;
                letter-spacing: 1px;
                text-transform: uppercase;
                text-shadow: 3px 3px 6px rgba(107, 70, 193, 0.3);
            '>
                Clusterflow
            </h1>
            <h3 style='
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-size: 1.3rem;
                font-weight: 400;
                color: #555;
                margin-top: 0;
                letter-spacing: 0.5px;
                font-style: italic;
            '>
                Discover which student group you belong to based on your gaming and music preferences!
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if models are loaded
    models_loaded = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This app uses machine learning to cluster students based on their:\n"
            "- Gaming preferences\n"
            "- Music preferences\n"
            "- Listening habits\n\n"
            "Fill in your preferences and click **Predict My Cluster** to see which group you belong to!"
        )
        
        st.header("🤖 Model Status")
        if models_loaded:
            st.success("✅ Models Loaded")
        else:
            st.error("❌ Models Not Loaded")
            st.warning("Please ensure model files are in the correct directory.")
        
        st.header("📊 Cluster Information")
        st.markdown("""
        **Cluster 0:** Casual Gamers & Bollywood Fans
        - Low listening hours
        - Prefer FPS & Sports games
        
        **Cluster 1:** Moderate Gamers & Pop Enthusiasts
        - Medium listening hours
        - Prefer Sports & Puzzle games
        
        **Cluster 2:** Hardcore Gamers & Music Lovers
        - High listening hours
        - Prefer Racing & FPS games
        """)
    
    # Main content
    if not models_loaded:
        st.error("⚠️ ML models are not loaded. Please check the model files.")
        return
    
    # Get available genres
    genres = get_available_genres()
    
    # Input Form
    st.header("📝 Enter Your Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎮 Gaming Preferences")
        game_1 = st.selectbox(
            "Top Game Genre",
            options=genres["game_genres"],
            help="Your most preferred game genre"
        )
        game_2 = st.selectbox(
            "Second Game Genre",
            options=genres["game_genres"],
            help="Your second most preferred game genre"
        )
        game_3 = st.selectbox(
            "Third Game Genre",
            options=genres["game_genres"],
            help="Your third most preferred game genre"
        )
    
    with col2:
        st.subheader("🎵 Music Preferences")
        music_1 = st.selectbox(
            "Top Music Genre",
            options=genres["music_genres"],
            help="Your most preferred music genre"
        )
        music_2 = st.selectbox(
            "Second Music Genre",
            options=genres["music_genres"],
            help="Your second most preferred music genre"
        )
        music_3 = st.selectbox(
            "Third Music Genre",
            options=genres["music_genres"],
            help="Your third most preferred music genre"
        )
    
    # Listening hours
    st.subheader("⏱️ Listening Habits")
    listening_hours = st.slider(
        "Daily Music Listening Hours",
        min_value=0,
        max_value=24,
        value=3,
        help="How many hours do you listen to music per day?"
    )
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 Predict My Cluster", use_container_width=True):
        # Prepare data
        input_data = {
            "game_genre_top_1": game_1,
            "game_genre_top_2": game_2,
            "game_genre_top_3": game_3,
            "music_genre_top_1": music_1,
            "music_genre_top_2": music_2,
            "music_genre_top_3": music_3,
            "listening_hours": listening_hours
        }
        
        # Show loading
        with st.spinner("🔄 Analyzing your preferences..."):
            result = make_prediction(input_data)
        
        # Display results
        if result["success"]:
            data = result["data"]
            
            st.success("✅ Prediction Complete!")
            
            # Main prediction box
            st.markdown("---")
            st.header("🎯 Your Cluster")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cluster Number", f"Cluster {data['cluster']}")
            with col2:
                st.metric("Cluster Size", f"{data['cluster_size']} students")
            with col3:
                st.metric("Confidence", data['confidence'])
            
            st.subheader(f"📌 {data['cluster_name']}")
            st.info(data['message'])
            
            # Detailed information
            st.markdown("---")
            st.header("📊 Cluster Profile")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎮 Top Games in Your Cluster")
                for i, game in enumerate(data['top_games'], 1):
                    st.write(f"{i}. **{game}**")
            
            with col2:
                st.subheader("🎵 Top Music in Your Cluster")
                for i, music in enumerate(data['top_music'], 1):
                    st.write(f"{i}. **{music}**")
            
            st.markdown("---")
            st.subheader("⏱️ Listening Habits")
            st.write(f"Average listening hours in your cluster: **{data['avg_listening_hours']} hours/day**")
            st.write(f"Your listening hours: **{listening_hours} hours/day**")
            
            # Comparison
            diff = listening_hours - data['avg_listening_hours']
            if abs(diff) < 1:
                st.success("✅ Your listening habits match your cluster perfectly!")
            elif diff > 0:
                st.info(f"ℹ️ You listen {abs(diff):.1f} hours more than your cluster average.")
            else:
                st.info(f"ℹ️ You listen {abs(diff):.1f} hours less than your cluster average.")
            
            # Download results
            st.markdown("---")
            st.download_button(
                label="📥 Download Results as JSON",
                data=json.dumps(data, indent=2),
                file_name="cluster_prediction.json",
                mime="application/json"
            )
            
        else:
            st.error(f"❌ Prediction Failed: {result['error']}")
            st.info("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
