import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from feature_extractor import extract_features
import os, warnings, json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────
DATASET  = "phishing_site_urls.csv"
SAMPLE_N = None  # Using full dataset (549k+ URLs) for maximum accuracy
OUT_DIR  = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def process_item(item):
    url, label = item
    try:
        feats = extract_features(url)
        if feats:
            feats["label"] = 1 if label == "bad" else 0
            return feats
    except:
        pass
    return None

if __name__ == "__main__":
    # ── 1. Load dataset ───────────────────────────────────────
    print("[1/5] Loading dataset...")
    df = pd.read_csv(DATASET)
    df.columns = df.columns.str.strip()
    df = df[["URL", "Label"]].dropna()
    df["Label"] = df["Label"].str.strip().str.lower()
    df = df[df["Label"].isin(["good", "bad"])]

    if SAMPLE_N:
        df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42).reset_index(drop=True)

    print(f"    Dataset : {len(df)} rows  |  bad={df['Label'].value_counts().get('bad', 0)}  good={df['Label'].value_counts().get('good', 0)}")

    # ── 2. Extract features (Parallel) ────────────────────────
    print("[2/5] Extracting features (Parallel)...")
    items = list(zip(df["URL"], df["Label"]))

    features_list = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_item, items, chunksize=1000), total=len(items), desc="Extracting features"):
            if result is not None:
                features_list.append(result)

    print(f"    Extracted features for {len(features_list)} URLs.")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    X = features_df.drop("label", axis=1)
    y = features_df["label"]

    # ── 3. Train / Test split ─────────────────────────────────
    print(f"[3/5] Training XGBoost on {len(X)} samples...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=800,
        max_depth=12,
        learning_rate=0.03,
        tree_method="hist",  # Faster for large datasets
        scale_pos_weight=2.5, # Handle imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Save model
    model.save_model(os.path.join(OUT_DIR, "phishing_model.json"))
    print(f"    Model saved -> {OUT_DIR}/phishing_model.json")

    # ── 4. Evaluate ───────────────────────────────────────────
    print("[4/5] Evaluating...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*45}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Phishing"],
                yticklabels=["Safe", "Phishing"])
    plt.title("Confusion Matrix - Phishing Detection")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
    print(f"    Saved -> {OUT_DIR}/confusion_matrix.png, {OUT_DIR}/roc_curve.png")

    # ── 5. Export Metadata for Dashboard ──────────────────────
    print("[5/5] Exporting metadata for dashboard...")
    
    # Feature Importance
    importances = model.feature_importances_
    feat_names = X.columns
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    
    # Sample predictions
    sample_indices = X_test.head(10).index
    sample_urls = df.loc[sample_indices, "URL"].tolist()
    sample_actual = y_test.head(10).tolist()
    sample_pred = y_pred[:10].tolist()
    
    samples = []
    for i in range(10):
        samples.append({
            "url": sample_urls[i],
            "actual": "phishing" if sample_actual[i] == 1 else "safe",
            "predicted": "phishing" if sample_pred[i] == 1 else "safe"
        })

    metadata = {
        "accuracy": round(acc * 100, 2),
        "total_test": int(len(y_test)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "metrics": {
            "safe": {
                "precision": round(report["0"]["precision"] * 100, 1),
                "recall": round(report["0"]["recall"] * 100, 1),
                "f1": round(report["0"]["f1-score"] * 100, 1),
                "support": int(report["0"]["support"])
            },
            "phishing": {
                "precision": round(report["1"]["precision"] * 100, 1),
                "recall": round(report["1"]["recall"] * 100, 1),
                "f1": round(report["1"]["f1-score"] * 100, 1),
                "support": int(report["1"]["support"])
            },
            "weighted_avg": {
                "precision": round(report["weighted avg"]["precision"] * 100, 1),
                "recall": round(report["weighted avg"]["recall"] * 100, 1),
                "f1": round(report["weighted avg"]["f1-score"] * 100, 1)
            }
        },
        "distribution": {
            "bad": int(df['Label'].value_counts().get('bad', 0)),
            "good": int(df['Label'].value_counts().get('good', 0))
        },
        "feature_importance": [{"name": n, "imp": float(i)} for n, i in feat_imp[:16]],
        "samples": samples
    }

    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"    Metadata saved -> {OUT_DIR}/metadata.json")
    print("\nDone! [OK]")
