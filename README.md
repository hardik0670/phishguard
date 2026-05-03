# PhishGuard: Phishing URL Detection System

PhishGuard is a high-performance phishing detection system that uses machine learning (XGBoost) to identify malicious URLs based on lexical and structural features. It features a real-time detection API and a comprehensive analytics dashboard.

## Features

- **XGBoost Classifier**: Trained on over 550,000 URLs for high precision and recall.
- **28+ Feature Extraction**: Analyzes URL length, entropy, brand mimicry, suspicious TLDs, and more.
- **Real-time API**: Flask-based backend for instant URL analysis.
- **Analytics Dashboard**: Visualizes model performance, feature importance, and live detection results.
- **Parallel Processing**: Fast feature extraction using multi-processing during training.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hardik0670/phishguard.git
   cd phishguard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
If you want to re-train the model on the full dataset:
```bash
python train.py
```
This will generate:
- `results/phishing_model.json` (The trained model)
- `results/metadata.json` (Model performance metrics and importance)
- `results/confusion_matrix.png` & `results/roc_curve.png`

### 2. Running the API & Dashboard
Start the Flask server:
```bash
python app.py
```
The dashboard will be available at `http://localhost:5000`.

## Project Structure

- `app.py`: Flask API and dashboard server.
- `train.py`: Model training and evaluation pipeline.
- `feature_extractor.py`: Logic for extracting 28+ features from a URL.
- `dashboard.html`: Single-page analytics dashboard.
- `phishing_site_urls.csv`: Dataset used for training.
- `results/`: Directory containing model artifacts and metadata.

## License
MIT
