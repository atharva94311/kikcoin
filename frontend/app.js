/**
 * TokenTrend Basic - Minimal Crypto Prediction Dashboard
 */

// Supported tokens
const TOKEN_MAP = {
  bitcoin: { symbol: "BTC", name: "Bitcoin" },
  ethereum: { symbol: "ETH", name: "Ethereum" },
  binancecoin: { symbol: "BNB", name: "BNB" },
  ripple: { symbol: "XRP", name: "XRP" },
  cardano: { symbol: "ADA", name: "Cardano" },
  solana: { symbol: "SOL", name: "Solana" },
  dogecoin: { symbol: "DOGE", name: "Dogecoin" },
  polkadot: { symbol: "DOT", name: "Polkadot" },
  chainlink: { symbol: "LINK", name: "Chainlink" },
  litecoin: { symbol: "LTC", name: "Litecoin" },
  stellar: { symbol: "XLM", name: "Stellar" },
  cosmos: { symbol: "ATOM", name: "Cosmos" },
  uniswap: { symbol: "UNI", name: "Uniswap" },
  aave: { symbol: "AAVE", name: "Aave" },
  eos: { symbol: "EOS", name: "EOS" },
  tron: { symbol: "TRX", name: "TRON" },
  iota: { symbol: "IOTA", name: "IOTA" },
  monero: { symbol: "XMR", name: "Monero" },
  nem: { symbol: "XEM", name: "NEM" },
  tether: { symbol: "USDT", name: "Tether" },
  "usd-coin": { symbol: "USDC", name: "USD Coin" },
  "crypto-com-chain": { symbol: "CRO", name: "Cronos" },
  "wrapped-bitcoin": { symbol: "WBTC", name: "Wrapped Bitcoin" }
};

const COINGECKO_API = "https://api.coingecko.com/api/v3";

let modelData = null;
let currentToken = "bitcoin";

// Basic formatting
function formatPrice(n) { return n ? "$" + n.toLocaleString("en-US", { maximumFractionDigits: 4 }) : "--"; }
function formatPct(n) { return n != null ? n.toFixed(2) + "%" : "--"; }
function formatCompact(n) {
  if (!n) return "--";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
  return "$" + n.toLocaleString();
}

// Update DOM elements
function updateDOM(id, text, colorClass = null) {
  const el = document.getElementById(id);
  if (el) {
    el.innerText = text;
    if (colorClass) {
      el.className = ""; // clear old
      if (colorClass !== "none") el.classList.add(colorClass);
    }
  }
}

const apiCache = {};
const CACHE_TTL = 60000; // 60 seconds

async function fetchWithCache(key, url) {
  const now = Date.now();
  if (apiCache[key] && now - apiCache[key].timestamp < CACHE_TTL) {
    return apiCache[key].data;
  }
  
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status);
  const data = await res.json();
  
  apiCache[key] = { timestamp: now, data };
  return data;
}

// 1. Fetch live market info
async function fetchMarketData(coinId) {
  try {
    const data = await fetchWithCache(`market_${coinId}`, `${COINGECKO_API}/coins/markets?vs_currency=usd&ids=${coinId}&price_change_percentage=24h`);
    return data[0];
  } catch (err) {
    console.error("Market fetch error", err);
    return null;
  }
}

// 2. Fetch historical data (needed for ML model features like RSI, Volume Ratio)
async function fetchHistoricalData(coinId) {
  try {
    return await fetchWithCache(`history_${coinId}`, `${COINGECKO_API}/coins/${coinId}/market_chart?vs_currency=usd&days=30&interval=daily`);
  } catch (err) {
    console.error("Historical fetch error", err);
    return null;
  }
}

// 3. Compute the exact features the ML Logistic Regression model expects
function computeFeatures(historicalData) {
  const prices = historicalData.prices.map(p => p[1]);
  const volumes = historicalData.total_volumes.map(v => v[1]);
  if (prices.length < 21) return null;

  const n = prices.length;
  const latest = n - 1;
  const dailyReturn = (prices[latest] - prices[latest - 1]) / prices[latest - 1];

  let sum7 = 0; for (let i = latest - 6; i <= latest; i++) sum7 += prices[i];
  const ma7 = sum7 / 7;

  let sum14 = 0; for (let i = latest - 13; i <= latest; i++) sum14 += prices[i];
  const ma14 = sum14 / 14;

  const returns14 = [];
  for (let i = latest - 13; i <= latest; i++) returns14.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  const mean14r = returns14.reduce((a, b) => a + b, 0) / 14;
  const variance14 = returns14.reduce((a, b) => a + Math.pow(b - mean14r, 2), 0) / 14;
  const volatility14d = Math.sqrt(variance14);

  const returns7 = returns14.slice(7);
  const mean7r = returns7.reduce((a, b) => a + b, 0) / 7;
  const variance7 = returns7.reduce((a, b) => a + Math.pow(b - mean7r, 2), 0) / 7;
  const volatility7d = Math.sqrt(variance7);

  let gains = 0, losses = 0;
  for (let i = latest - 13; i <= latest; i++) {
    let diff = prices[i] - prices[i - 1];
    if (diff > 0) gains += diff; else losses -= diff;
  }
  const rs = (losses === 0) ? 100 : (gains / 14) / (losses / 14);
  const rsi14 = 100 - (100 / (1 + rs));

  let volSum7 = 0; for (let i = latest - 6; i <= latest; i++) volSum7 += volumes[i];
  const volMa7 = volSum7 / 7;

  return {
    daily_return: dailyReturn,
    price_to_ma7: prices[latest] / ma7,
    price_to_ma14: prices[latest] / ma14,
    ma7_to_ma14: ma7 / ma14,
    volatility_7d: volatility7d,
    volatility_14d: volatility14d,
    momentum_7d: (prices[latest] - prices[latest - 7]) / prices[latest - 7],
    momentum_14d: (prices[latest] - prices[latest - 14]) / prices[latest - 14],
    rsi_14: rsi14,
    volume_change: (volumes[latest] - volumes[latest - 1]) / volumes[latest - 1],
    volume_ratio: volumes[latest] / volMa7
  };
}

// 4. Run the Machine Learning Inference
function predict(features, tokenModel, featureNames) {
  let z = tokenModel.bias;
  for (let i = 0; i < featureNames.length; i++) {
    const f = featureNames[i];
    const val = features[f] || 0;
    const std = tokenModel.scaler_std[i];
    const scaled = std === 0 ? 0 : (val - tokenModel.scaler_mean[i]) / std;
    z += tokenModel.weights[i] * scaled;
  }
  
  const probability = 1 / (1 + Math.exp(-z)); // sigmoid
  const direction = probability >= 0.5 ? "UP" : "DOWN";
  const confidence = direction === "UP" ? probability : 1 - probability;
  
  let signal = "Neutral";
  if (confidence >= 0.75) signal = direction === "UP" ? "Strong Buy" : "Strong Sell";
  else if (confidence >= 0.60) signal = direction === "UP" ? "Buy" : "Sell";
  
  return { direction, confidence, signal };
}

// Main logic
async function loadAndRefresh() {
  updateDOM("status", "Loading data...");
  document.getElementById("dataPanel").style.display = "none";

  // Load Model JSON if not loaded
  if (!modelData) {
    try {
      const res = await fetch("data/model_data.json");
      modelData = await res.json();
    } catch(e) {
      updateDOM("status", "Error loading AI model data - have you trained it?");
      return;
    }
  }

  // Fetch API data
  const market = await fetchMarketData(currentToken);
  const history = await fetchHistoricalData(currentToken);

  if (!market || !history) {
    updateDOM("status", "Error fetching data from CoinGecko API. Try again in a minute.");
    return;
  }

  // Update UI Stats
  updateDOM("priceVal", formatPrice(market.current_price));
  updateDOM("changeVal", formatPct(market.price_change_percentage_24h), market.price_change_percentage_24h >= 0 ? "text-up" : "text-down");
  updateDOM("mcapVal", formatCompact(market.market_cap));
  updateDOM("volumeVal", formatCompact(market.total_volume));

  // Run AI AI Prediction
  const features = computeFeatures(history);
  const symbol = TOKEN_MAP[currentToken].symbol;
  const tokenModel = modelData.tokens[symbol];

  if (features && tokenModel) {
    const result = predict(features, tokenModel, modelData.features);
    updateDOM("predDir", result.direction, result.direction === "UP" ? "text-up" : "text-down");
    updateDOM("predConf", (result.confidence * 100).toFixed(1) + "%");
    updateDOM("predSignal", result.signal);
  } else {
    updateDOM("predDir", "Model not trained", "none");
    updateDOM("predConf", "--", "none");
    updateDOM("predSignal", "--", "none");
  }

  document.getElementById("dataPanel").style.display = "block";
  updateDOM("status", "Data loaded and predictions updated.");
}

// Initialize App
window.onload = () => {
  const select = document.getElementById("tokenSelect");
  
  // Create dropdown options
  for (const [id, info] of Object.entries(TOKEN_MAP)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.innerText = `${info.name} (${info.symbol})`;
    select.appendChild(opt);
  }
  
  // Handlers
  select.onchange = (e) => { currentToken = e.target.value; loadAndRefresh(); };
  document.getElementById("refreshBtn").onclick = loadAndRefresh;
  
  // Initial load
  loadAndRefresh();
};
