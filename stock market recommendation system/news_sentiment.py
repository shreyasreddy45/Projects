# news_sentiment.py

"""
News + Sentiment module using GNews for fetching headlines
and VADER/FinBERT for sentiment scoring.

This version is the MOST stable and recommended for your project.
"""

from gnews import GNews
from typing import List, Dict

# ======================
# VADER Sentiment
# ======================

def install_nltk_vader_if_missing():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except:
        import nltk
        nltk.download('vader_lexicon')

def vader_sentiment_score(texts: List[str]) -> float:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    scores = []
    for t in texts:
        if not isinstance(t, str) or t.strip() == "":
            continue
        s = analyzer.polarity_scores(t)
        scores.append(s["compound"])

    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))


# ======================
# FinBERT Sentiment (optional)
# ======================

def hf_finbert_sentiment_score(texts: List[str], model_name="yiyanghkust/finbert-tone") -> float:
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    scores = []
    for t in texts:
        if not t.strip():
            continue

        out = pipe(t[:512])[0]
        label = out["label"].lower()
        score = out["score"]

        if "positive" in label:
            scores.append(score)
        elif "negative" in label:
            scores.append(-score)
        else:
            scores.append(0.0)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ======================
# Fetch Headlines (GNews)
# ======================

def fetch_google_news_headlines(query: str, max_results: int = 8) -> List[Dict]:
    """
    Using GNews library to fetch reliable headlines.
    Returns list of dicts with: title, link, source
    """
    google = GNews(language='en', country='IN', max_results=max_results)

    raw = google.get_news(query)
    results = []

    for r in raw:
        try:
            results.append({
                "title": r.get("title", ""),
                "link": r.get("url", ""),
                "source": r.get("publisher", {}).get("title", "Unknown")
            })
        except:
            continue

    return results


# ======================
# Aggregation
# ======================

def aggregate_sentiment(texts: List[str], method="vader") -> float:
    if method == "vader":
        install_nltk_vader_if_missing()
        return vader_sentiment_score(texts)
    elif method in ("finbert", "hf-finbert"):
        return hf_finbert_sentiment_score(texts)
    else:
        raise ValueError("Unknown sentiment method")


# ======================
# Main Utility
# ======================

def get_company_sentiment(symbol: str, method="vader", max_results=8) -> Dict:
    headlines = fetch_google_news_headlines(symbol, max_results=max_results)

    titles = [h["title"] for h in headlines if h.get("title")]
    score = aggregate_sentiment(titles, method=method)

    return {
        "symbol": symbol,
        "sentiment_score": score,
        "headlines": headlines,
        "method": method
    }


# Debug Testing
if __name__ == "__main__":
    out = get_company_sentiment("RELIANCE", method="vader")
    print("Sentiment:", out["sentiment_score"])
    for h in out["headlines"]:
        print("-", h["title"])
