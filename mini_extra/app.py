from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# === Import Sentiment Module ===
from news_sentiment import get_company_sentiment

app = Flask(__name__)
CORS(app)


# ============================
# Load All Models and Scalers
# ============================

models = {}
scalers = {}

for file in os.listdir("models"):
    if file.endswith("_model.pkl"):
        symbol = file.replace("_model.pkl", "")
        models[symbol] = joblib.load(f"models/{symbol}_model.pkl")
        scalers[symbol] = joblib.load(f"models/{symbol}_scaler.pkl")

print("Loaded models:", list(models.keys()))


# ============================
# Indicators (match training)
# ============================

def add_indicators(df):
    df['Returns_1'] = df['Close'].pct_change(1)
    df['Returns_5'] = df['Close'].pct_change(5)

    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Price_vs_MA10'] = (df['Close'] / df['MA_10']) - 1
    df['Price_vs_MA20'] = (df['Close'] / df['MA_20']) - 1

    df['Volatility'] = df['Close'].pct_change().rolling(10).std()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Volume_MA'] = df['Volume'].rolling(10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    return df.dropna()


# ============================
# Prediction + Sentiment Route
# ============================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symbol = data.get("symbol")
    sentiment_method = data.get("sentiment_method", "vader")

    # --- Validate Symbol ---
    if symbol not in models:
        return jsonify({"error": "Model for this stock not found"}), 400

    file_path = f"stock_data/{symbol}.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "CSV not found"}), 400

    # --- Load Stock Data ---
    df = pd.read_csv(file_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    # --- Indicators ---
    df = add_indicators(df)
    latest = df.iloc[-1:]

    feature_cols = [
        col for col in latest.columns
        if col not in ["Signal"] and latest[col].dtype != object
    ]

    X = latest[feature_cols].values
    X_scaled = scalers[symbol].transform(X)
    model_pred = models[symbol].predict(X_scaled)[0]

    model_label = {0: "SELL", 1: "HOLD", 2: "BUY"}[model_pred]

    # Map model output → numeric
    model_numeric = -1 if model_pred == 0 else (0 if model_pred == 1 else 1)

    # ================================
    # Fetch Sentiment (News + NLP)
    # ================================
    sentiment = get_company_sentiment(symbol, method=sentiment_method, max_results=6)
    sentiment_score = sentiment["sentiment_score"]  # -1 to +1

    # ================================
    # Combine Scores to Final Output
    # ================================
    final_score = 0.6 * model_numeric + 0.4 * sentiment_score

    if final_score >= 0.4:
        final_label = "BUY"
    elif final_score <= -0.4:
        final_label = "SELL"
    else:
        final_label = "HOLD"

    # ================================
    # Explainability Section
    # ================================

    explanation = {
        "model_prediction": model_label,
        "sentiment_score": sentiment_score,
        "final_score": final_score,
        "why": []
    }

    # Explain based on sentiment
    if sentiment_score > 0.2:
        explanation["why"].append("News sentiment is positive.")
    elif sentiment_score < -0.2:
        explanation["why"].append("News sentiment is negative.")
    else:
        explanation["why"].append("News sentiment is neutral.")

    # Explain based on ML model
    explanation["why"].append(f"ML model suggests: {model_label}.")
    explanation["why"].append(f"Combined score = {final_score:.3f} → {final_label}.")

    # ================================
    # Return JSON
    # ================================

    return jsonify({
        "symbol": symbol,
        "prediction": final_label,
        "model_prediction": model_label,
        "sentiment": sentiment,
        "final_score": final_score,
        "explanation": explanation
    })

@app.route("/top5", methods=["GET"])
def top5():
    results = []

    for symbol in models.keys():
        try:
            # Load CSV
            file_path = f"stock_data/{symbol}.csv"
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)

            # Indicators
            df = add_indicators(df)
            latest = df.iloc[-1:]

            feature_cols = [
                col for col in latest.columns
                if col not in ["Signal"] and latest[col].dtype != object
            ]

            X = latest[feature_cols].values
            X_scaled = scalers[symbol].transform(X)

            pred = models[symbol].predict(X_scaled)[0]
            model_numeric = -1 if pred == 0 else (0 if pred == 1 else 1)

            # Sentiment
            query_name = symbol.replace("_", " ")
            sentiment = get_company_sentiment(query_name, method="vader", max_results=3)
            sentiment_score = sentiment["sentiment_score"]

            # Combined score
            final_score = 0.6 * model_numeric + 0.4 * sentiment_score

            results.append({
                "symbol": symbol,
                "final_score": final_score
            })

        except:
            continue

    # Sort high → low
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # Send top 5
    return jsonify(results[:5])



# ============================
# Run Backend
# ============================

if __name__ == "__main__":
    app.run(debug=True)
