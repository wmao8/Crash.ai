⚠️ Note: Due to GitHub file size limits, the dataset `US_Accidents_March23_sampled_500k.csv` is not included in this repository.  
Please download it manually from https://drive.google.com/file/d/1U3u8QYzLjnEaSurtZfSAS_oh9AT2Mn8X/edit and place it in the project root folder.

# U.S. Accident Severity Prediction App (Streamlit + Docker)

This project is a Streamlit-based web application that predicts the severity of traffic accidents in the U.S. using real-time environmental features. It includes a pre-trained histgradientboost model, interactive visualizations, and a full Dockerized setup for quick deployment.

## 🔍 Features

- 🌐 Interactive web UI built with Streamlit
- 📊 Visual analytics of accident data (weather, location, time)
- 🤖 ML model prediction (Histgradientboost Model)
- 📦 One-command Docker deployment
- 📁 Sample dataset and model included

## 📁 Project Structure
<pre>. ├── streamlit_app_US_accidents.py # Main app file
├── requirements.txt # Python dependencies
├── Dockerfile # Docker container setup
├── histgradientboost_model.pkl # Pre-trained ML model
├── US_Accidents_March23_sampled_500k.csv # Input dataset (manually downloaded)
</pre>

## 🚀 Getting Started

### Option 1: Run locally (Python environment required)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run streamlit_app_US_accidents.py
```

### Option 2: Run via Docker (recommended)

1. Build the Docker image:

```bash
docker build -t us-accident-app .
```

2. Run the container:

 ```bash
docker run -p 8501:8501 us-accident-app
```

Then open http://localhost:8501 in your browser.
