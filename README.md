# ML Stock Predictor

An LSTM (Long Short-Term Memory) neural network that predicts Apple Inc. (AAPL) closing stock prices using 60 days of historical price data.

---

## How It Works

1. Downloads historical AAPL data from Yahoo Finance (2012–2024)
2. Scales prices to a 0–1 range using MinMaxScaler
3. Trains a 2-layer LSTM model on 80% of the data
4. Evaluates on the remaining 20% and reports RMSE
5. Predicts the next trading day's closing price
6. Plots training data, actual prices, and predicted prices

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/jonowrenn/MLStockPredictor.git
cd MLStockPredictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python Stock.py
```

Training takes a few minutes. Three matplotlib charts will display: the full price history, and the actual vs. predicted price comparison.

---

## Tech Stack

- Python
- TensorFlow / Keras — LSTM model
- yfinance — stock data
- scikit-learn — data scaling
- pandas / numpy — data processing
- matplotlib — visualization
