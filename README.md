# TokenTrend — AI-Powered Crypto Prediction Dashboard

> An intelligent, web-based crypto tracking dashboard that uses machine learning to predict short-term market trends and explain its reasoning.

![Dashboard Preview](frontend/screenshot.png)

---

## 🚀 Features

| Feature | Description |
|---|---|
| **AI Predictions** | Logistic Regression model predicts next-day UP/DOWN for 15 tokens |
| **Explainable AI** | Shows *why* the model predicts — top feature contributions with Bullish/Bearish labels |
| **Signal Strength** | Strong Buy → Neutral → Strong Sell based on confidence thresholds |
| **Live Data** | CoinGecko API integration for real-time prices, volume, market cap |
| **Client-Side Inference** | Model weights exported to JSON — predictions run in the browser, no backend needed |
| **Multi-Token Ranking** | All tokens ranked by prediction signal strength |
| **Interactive Charts** | Chart.js price charts with 7D/14D/30D/90D/1Y range selection |
| **Premium UI** | Dark theme with glassmorphism, gradient accents, shimmer loading, responsive |

---

## 🏗️ Architecture

```
Python Pipeline (offline)              Frontend (browser, real-time)
┌──────────────────────┐              ┌──────────────────────────┐
│ Historical Data      │              │ CoinGecko API (live)     │
│        ↓             │              │        ↓                 │
│ Feature Engineering  │              │ Feature Computation (JS) │
│   - MA7, MA14        │              │        ↓                 │
│   - RSI(14)          │  ─────────►  │ Sigmoid Inference        │
│   - Volatility       │ model_data   │   z = w·x + b           │
│   - Momentum         │   .json      │   P = 1/(1+e^-z)        │
│        ↓             │              │        ↓                 │
│ Logistic Regression  │              │ Explainability Engine    │
│        ↓             │              │        ↓                 │
│ Export weights+scaler │              │ Dashboard UI (Chart.js)  │
└──────────────────────┘              └──────────────────────────┘
```

---

## 📊 ML Pipeline

### Features Engineered (11 total)
| Feature | Category | Description |
|---|---|---|
| `daily_return` | Price | Day-over-day price change % |
| `price_to_ma7` | Trend | Price relative to 7-day moving average |
| `price_to_ma14` | Trend | Price relative to 14-day moving average |
| `ma7_to_ma14` | Trend | Short-term vs long-term trend (crossover) |
| `volatility_7d` | Risk | 7-day rolling standard deviation of returns |
| `volatility_14d` | Risk | 14-day rolling standard deviation of returns |
| `momentum_7d` | Momentum | 7-day price change % |
| `momentum_14d` | Momentum | 14-day price change % |
| `rsi_14` | Oscillator | Relative Strength Index (overbought/oversold) |
| `volume_change` | Volume | Day-over-day volume change % |
| `volume_ratio` | Volume | Current volume vs 7-day average |

### Model Choice: Logistic Regression
**Why?**
- **Interpretable**: Weights directly map to feature importance → enables explainability
- **Lightweight**: Exports as a simple `{weights, bias}` JSON — trivial to run in JavaScript
- **Fast inference**: Single dot product + sigmoid — runs in <1ms in the browser
- **Balanced classes**: Uses `class_weight="balanced"` to handle UP/DOWN imbalance
- **No overfitting**: Simple model avoids overfitting on noisy financial data

### Client-Side Inference
The model runs entirely in the browser:
```javascript
function predict(features, weights, bias) {
  let z = bias;
  for (let i = 0; i < features.length; i++) {
    z += features[i] * weights[i];
  }
  return 1 / (1 + Math.exp(-z)); // sigmoid → probability
}
```

Features are scaled using saved StandardScaler parameters (`mean`, `std`) from training.

---

## 🛠️ Setup & Running

### Prerequisites
- Python 3.8+
- pip packages: `pandas`, `numpy`, `scikit-learn`, `requests`

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn requests
```

### 2. Train the ML Model
```bash
# Using synthetic data (built-in)
python ml/generate_data.py
python ml/train_model.py ml/datasets

# OR using your own CSV files
python ml/train_model.py /path/to/your/csvs
```

The pipeline outputs `frontend/data/model_data.json` with weights for all tokens.

### 3. Launch the Dashboard
```bash
cd frontend
python -m http.server 8000
```
Open http://localhost:8000

---

## 📁 Project Structure
```
kikcoin/
├── ml/
│   ├── data_pipeline.py      # Feature engineering & data loading
│   ├── train_model.py        # Model training & JSON export
│   ├── generate_data.py      # Synthetic data generator (GBM)
│   └── datasets/             # CSV data files
├── frontend/
│   ├── index.html            # Dashboard HTML
│   ├── style.css             # Premium dark theme CSS
│   ├── app.js                # Client-side inference & UI
│   └── data/
│       └── model_data.json   # Exported model weights
└── README.md
```

---

## 🎯 Supported Tokens
BTC, ETH, SOL, BNB, XRP, ADA, DOGE, DOT, AVAX, LINK, MATIC, LTC, UNI, XLM, ATOM

---

## 🔮 Innovation Highlights

1. **Explainability Panel**: Shows the top 5 feature contributions behind each prediction with human-readable labels and bullish/bearish classification
2. **Two-Phase Prediction**: Pre-computed predictions load instantly from model JSON; live CoinGecko data upgrades them asynchronously when available
3. **Signal Strength Meter**: Maps confidence to actionable trading signals (Strong Buy → Neutral → Strong Sell)
4. **Multi-Token Ranking**: All tokens ranked by predicted signal strength — shows "top gainers" at a glance

---

## 👥 Team
KikCoin

---

## 📜 License
MIT
