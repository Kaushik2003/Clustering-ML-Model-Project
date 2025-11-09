# 🎮 Clusterflow - Student Clustering ML Application

A full-stack machine learning application that clusters students based on their gaming and music preferences using K-Prototypes algorithm.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **Interactive Web Interface** - Beautiful grey liquid glass themed UI built with Streamlit
- **Real-time Predictions** - Instant cluster predictions based on user preferences
- **RESTful API** - FastAPI backend for scalable predictions
- **ML-Powered** - K-Prototypes clustering algorithm for mixed data types
- **Cloud Deployed** - Backend on Render, Frontend on Streamlit Cloud

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Streamlit     │  HTTP   │   FastAPI        │  Load   │   ML Models     │
│   Frontend      │ ──────> │   Backend        │ ──────> │   (.joblib)     │
│  (Cloud)        │         │   (Render)       │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

## 📊 Dataset

The model is trained on student data including:
- **Gaming Preferences**: Top 3 favorite game genres (FPS, Sports, Racing, etc.)
- **Music Preferences**: Top 3 favorite music genres (Pop, Rock, Hip-Hop, etc.)
- **Listening Habits**: Daily music listening hours

## 🚀 Live Demo

- **Frontend**: [https://clusterflow.streamlit.app](https://clusterflow.streamlit.app)
- **API Docs**: [https://your-render-url.onrender.com/docs](https://your-render-url.onrender.com/docs)

## 🛠️ Tech Stack

### Frontend
- **Streamlit** - Interactive web application
- **Requests** - API communication
- **Custom CSS** - Grey liquid glass theme

### Backend
- **FastAPI** - High-performance API framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Machine Learning
- **scikit-learn** - Data preprocessing
- **kmodes** - K-Prototypes clustering
- **joblib** - Model serialization
- **NumPy** - Numerical operations

## 📁 Project Structure

```
Clustering-ML-Model-Project/
├── Clusterflow/
│   ├── backend/
│   │   ├── main.py              # FastAPI application
│   │   └── requirements.txt     # Backend dependencies
│   ├── frontend/
│   │   └── app.py              # Streamlit application
│   └── model/
│       ├── kprototypes_model.joblib
│       ├── ordinal_encoder.joblib
│       ├── standard_scaler.joblib
│       └── model_info.joblib
├── notebook/
│   ├── dataset.csv             # Training data
│   └── run_clustering.py       # Model training script
├── figures/                    # Visualizations
├── requirements.txt            # Project dependencies
└── render.yaml                # Render deployment config
```

## 🔧 Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Sippy063/Clustering-ML-Model-Project.git
cd Clustering-ML-Model-Project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running Locally

**Start Backend (Terminal 1)**
```bash
cd Clusterflow/backend
python main.py
```
Backend will run on `http://localhost:8000`

**Start Frontend (Terminal 2)**
```bash
cd Clusterflow/frontend
streamlit run app.py
```
Frontend will run on `http://localhost:8501`

## ☁️ Deployment

### Backend (Render)

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `pip install fastapi uvicorn[standard] pydantic joblib numpy scikit-learn kmodes`
   - **Start Command**: `cd Clusterflow/backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy!

### Frontend (Streamlit Cloud)

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Deploy from GitHub repository
3. Set main file path: `Clusterflow/frontend/app.py`
4. Add secret in Settings → Secrets:
```toml
API_URL = "https://your-render-url.onrender.com"
```
5. Deploy!

## 📡 API Endpoints

### Health Check
```http
GET /health
```
Returns API health status

### Get Available Genres
```http
GET /available-genres
```
Returns list of available game and music genres

### Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "game_genre_top_1": "FPS",
  "game_genre_top_2": "Sports",
  "game_genre_top_3": "Racing",
  "music_genre_top_1": "Pop",
  "music_genre_top_2": "Rock",
  "music_genre_top_3": "Hip-Hop",
  "listening_hours": 5
}
```

## 🎯 Clusters

The model identifies 3 distinct student clusters:

- **Cluster 0**: Casual Gamers & Bollywood Fans
  - Low listening hours
  - Prefer FPS & Sports games

- **Cluster 1**: Moderate Gamers & Pop Enthusiasts
  - Medium listening hours
  - Prefer Sports & Puzzle games

- **Cluster 2**: Hardcore Gamers & Music Lovers
  - High listening hours
  - Prefer Racing & FPS games

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sippy063**
- GitHub: [@Sippy063](https://github.com/Sippy063)

## 🙏 Acknowledgments

- K-Prototypes algorithm for handling mixed data types
- Streamlit for the amazing web framework
- FastAPI for the high-performance backend
- Render for free hosting

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ Star this repo if you find it helpful!
