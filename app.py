from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from xgboost import XGBClassifier
import pandas as pd
import os
from urllib.parse import urlparse
from feature_extractor import detect_brand_risk, extract_features

app = Flask(__name__)
CORS(app)

# ── Configuration ──────────────────────────────────────────
MODEL_PATH = "results/phishing_model.json"
METADATA_PATH = "results/metadata.json"

def is_valid_url_input(url):
    candidate = url.strip()
    if any(char.isspace() for char in candidate):
        return False
    parsed = urlparse(candidate if candidate.startswith(("http://", "https://")) else "http://" + candidate)
    host = (parsed.hostname or "").strip(".")
    if not host or "." not in host:
        return False
    labels = host.split(".")
    tld = labels[-1]
    return all(labels) and len(tld) >= 2 and tld.isalpha()

# Load the model
model = XGBClassifier()
model_loaded = False
if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    model_loaded = True
    print(f"[*] Model loaded from {MODEL_PATH}")
else:
    print(f"[!] Model not found at {MODEL_PATH}. Run train.py first.")

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory("results", filename)

@app.route("/health")
def health():
    return jsonify({
        "app": "PhishGuard API",
        "status": "active",
        "model_loaded": model_loaded
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not is_valid_url_input(url):
        return jsonify({"error": "Enter a valid URL or domain, for example https://paypal.com"}), 400
    
    feats = extract_features(url)
    if not feats:
        return jsonify({"error": "Could not extract features from URL"}), 400

    if not model_loaded:
        return jsonify({"error": "Model is not loaded"}), 503
    
    # Convert to DataFrame to ensure correct column order (same as training)
    df_feats = pd.DataFrame([feats])
    
    # Get prediction and probability
    pred = int(model.predict(df_feats)[0])
    prob = float(model.predict_proba(df_feats)[0][1])
    brand_risk = detect_brand_risk(url)

    # Domain-aware guardrails catch real-world brand abuse the model can miss.
    if brand_risk["suspicious_brand"]:
        pred = 1
        prob = max(prob, 0.92)
    elif brand_risk["official_brand"] and not any([
        feats["has_ip"],
        feats["at_count"] > 0,
        feats["has_suspicious_tld"],
        feats["has_redirection"],
        feats["tld_in_path"],
    ]):
        pred = 0
        prob = min(prob, 0.08)
    
    # Identify triggered flags (heuristic-like explanations)
    flags = []
    if feats["has_https"] == 0: flags.append("Insecure connection (HTTP)")
    if feats["has_phishing_words"]: flags.append("Phishing keywords detected")
    if feats["has_ip"]: flags.append("IP address in domain")
    if feats["has_suspicious_tld"]: flags.append("Suspicious TLD")
    if feats["entropy"] > 4.5: flags.append("Unusually high character entropy")
    if brand_risk["suspicious_brand"]: flags.append(f"Brand lookalike: {brand_risk['suspicious_reason']}")
    if feats["is_shortened"]: flags.append("URL shortener used")
    if feats["at_count"] > 0: flags.append("@ symbol present")
    if feats["has_redirection"]: flags.append("Internal redirection detected")
    
    return jsonify({
        "url": url,
        "verdict": "phishing" if pred == 1 else "safe",
        "score": round(prob * 100, 2),
        "flags": flags,
        "features": feats,
        "brand_risk": brand_risk
    })

if __name__ == "__main__":
    print("[*] PhishGuard API starting on http://localhost:5000")
    app.run(port=5000, debug=True)
