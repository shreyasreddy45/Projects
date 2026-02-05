import pandas as pd
import numpy as np
import os
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
import joblib
import glob

warnings.filterwarnings('ignore')


class StockPredictor:
    def __init__(self, data_folder_path="stock_data"):
        self.models = {}
        self.scalers = {}
        self.stock_data = {}
        self.results = {}
        self.data_folder_path = data_folder_path

    def load_data(self, symbol):
        """Load stock data from CSV"""
        try:
            file_path = os.path.join(self.data_folder_path, f"{symbol}.csv")
            df = pd.read_csv(file_path)

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)

            # Convert numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            self.stock_data[symbol] = df
            print(f"Loaded {symbol}: {len(df)} rows")
            return df

        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None

    def add_indicators(self, df):
        """Add technical indicators"""
        df['Returns_1'] = df['Close'].pct_change(1)
        df['Returns_5'] = df['Close'].pct_change(5)

        df['MA_10'] = df['Close'].rolling(10).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['Price_vs_MA10'] = (df['Close'] / df['MA_10']) - 1
        df['Price_vs_MA20'] = (df['Close'] / df['MA_20']) - 1

        df['Volatility'] = df['Close'].pct_change().rolling(10).std()

        # RSI (fixed)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        return df.dropna()

    def create_signals(self, df):
        """Create BUY / SELL / HOLD signals"""
        future_return = (df['Close'].shift(-3) / df['Close']) - 1

        df['Signal'] = np.where(
            future_return > 0.02,  # buy
            2,
            np.where(
                future_return < -0.02,  # sell
                0,
                1  # hold
            )
        )

        return df.dropna()

    def train_model(self, symbol):
        """Train the ML model"""
        try:
            df = self.stock_data[symbol].copy()
            df = self.add_indicators(df)
            df = self.create_signals(df)

            feature_cols = [
                col for col in df.columns 
                if col not in ['Signal'] and df[col].dtype != object
            ]

            X = df[feature_cols].values
            y = df['Signal'].values

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[symbol] = scaler

            model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    num_class=3,
                    eval_metric='mlogloss'
            )
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            self.models[symbol] = model
            self.results[symbol] = {
                "accuracy": accuracy,
                "report": classification_report(
                    y_test, y_pred,
                    target_names=['SELL', 'HOLD', 'BUY']
                ),
                "actual": y_test,
                "predicted": y_pred
            }

            print(f"Model trained for {symbol} - Accuracy: {accuracy:.2%}")
            return accuracy

        except Exception as e:
            print(f"Training error for {symbol}: {e}")
            return 0

    def predict_current_signal(self, symbol):
        """Predict last day's BUY/SELL/HOLD"""
        if symbol not in self.models:
            print("Model not trained for this stock")
            return None

        df = self.stock_data[symbol].copy()
        df = self.add_indicators(df)

        latest = df.iloc[-1:]

        feature_cols = [
            col for col in latest.columns
            if col not in ['Signal'] and latest[col].dtype != object
        ]

        X = latest[feature_cols].values
        X_scaled = self.scalers[symbol].transform(X)

        pred = self.models[symbol].predict(X_scaled)[0]

        return {0: "SELL", 1: "HOLD", 2: "BUY"}[pred]


# ========================================================================
# RUN SCRIPT
# ========================================================================
if __name__ == "__main__":

    predictor = StockPredictor("stock_data")

    if not os.path.exists("models"):
        os.makedirs("models")

    csv_files = glob.glob("stock_data/*.csv")

    for path in csv_files:
        symbol = os.path.basename(path).replace(".csv", "")

        print("\n===================================================")
        print(f" Training model for: {symbol}")
        print("===================================================")

        df = predictor.load_data(symbol)
        if df is None:
            continue

        acc = predictor.train_model(symbol)

        if acc > 0:
            joblib.dump(predictor.models[symbol], f"models/{symbol}_model.pkl")
            joblib.dump(predictor.scalers[symbol], f"models/{symbol}_scaler.pkl")
            print(f"Model + scaler saved for {symbol}")
