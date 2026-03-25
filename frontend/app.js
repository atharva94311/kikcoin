/**
 * TokenTrend - AI-Powered Crypto Prediction Dashboard
 *
 * Client-side inference engine: loads model weights from JSON,
 * fetches live data from CoinGecko, computes features, and
 * runs logistic regression prediction in the browser.
 */

// == Token Registry ==
const TOKEN_MAP = {
  bitcoin:              { symbol: "BTC",   name: "Bitcoin",         color: "#f7931a" },
  ethereum:             { symbol: "ETH",   name: "Ethereum",        color: "#627eea" },
  binancecoin:          { symbol: "BNB",   name: "BNB",             color: "#f3ba2f" },
  ripple:               { symbol: "XRP",   name: "XRP",             color: "#00aae4" },
  cardano:              { symbol: "ADA",   name: "Cardano",         color: "#0033ad" },
  solana:               { symbol: "SOL",   name: "Solana",          color: "#9945ff" },
  dogecoin:             { symbol: "DOGE",  name: "Dogecoin",        color: "#c2a633" },
  polkadot:             { symbol: "DOT",   name: "Polkadot",        color: "#e6007a" },
  chainlink:            { symbol: "LINK",  name: "Chainlink",       color: "#2a5ada" },
  litecoin:             { symbol: "LTC",   name: "Litecoin",        color: "#bfbbbb" },
  stellar:              { symbol: "XLM",   name: "Stellar",         color: "#14b6e7" },
  cosmos:               { symbol: "ATOM",  name: "Cosmos",          color: "#2e3148" },
  uniswap:              { symbol: "UNI",   name: "Uniswap",         color: "#ff007a" },
  aave:                 { symbol: "AAVE",  name: "Aave",            color: "#b6509e" },
  eos:                  { symbol: "EOS",   name: "EOS",             color: "#443f54" },
  tron:                 { symbol: "TRX",   name: "TRON",            color: "#ff0013" },
  iota:                 { symbol: "IOTA",  name: "IOTA",            color: "#131f37" },
  monero:               { symbol: "XMR",   name: "Monero",          color: "#ff6600" },
  nem:                  { symbol: "XEM",   name: "NEM",             color: "#67b2e8" },
  tether:               { symbol: "USDT",  name: "Tether",          color: "#26a17b" },
  "usd-coin":           { symbol: "USDC",  name: "USD Coin",        color: "#2775ca" },
  "crypto-com-chain":   { symbol: "CRO",   name: "Cronos",          color: "#002d74" },
  "wrapped-bitcoin":    { symbol: "WBTC",  name: "Wrapped Bitcoin", color: "#f09242" },
};

const COINGECKO_API = "https://api.coingecko.com/api/v3";

// == App State ==
const state = {
  selectedToken: "bitcoin",
  chartDays: 7,
  modelData: null,
  marketData: {},
  chartData: {},
  predictions: {},
  priceChart: null,
  isLoading: false,
};

// == Utility Functions ==
function formatPrice(n) {
  if (n == null) return "\u2014";
  if (n >= 1) return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 0.01) return "$" + n.toFixed(4);
  return "$" + n.toFixed(6);
}

function formatCompact(n) {
  if (n == null) return "\u2014";
  if (n >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return "$" + (n / 1e3).toFixed(1) + "K";
  return "$" + n.toFixed(2);
}

function formatPct(n) {
  if (n == null) return "\u2014";
  var sign = n >= 0 ? "+" : "";
  return sign + n.toFixed(2) + "%";
}

function showToast(msg) {
  var el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(function() { el.classList.remove("show"); }, 3000);
}

// == CoinGecko API ==
async function fetchMarketData() {
  var ids = Object.keys(TOKEN_MAP).join(",");
  var url = COINGECKO_API + "/coins/markets?vs_currency=usd&ids=" + ids + "&order=market_cap_desc&sparkline=true&price_change_percentage=24h,7d";
  var resp = await fetch(url);
  if (!resp.ok) throw new Error("Markets API: " + resp.status);
  var data = await resp.json();
  for (var i = 0; i < data.length; i++) {
    state.marketData[data[i].id] = data[i];
  }
  return data;
}

async function fetchChartData(coinId, days) {
  days = days || 7;
  var url = COINGECKO_API + "/coins/" + coinId + "/market_chart?vs_currency=usd&days=" + days;
  var resp = await fetch(url);
  if (!resp.ok) throw new Error("Chart API: " + resp.status);
  var data = await resp.json();
  state.chartData[coinId] = data;
  return data;
}

// == Feature Engineering (JS) ==
function computeFeatures(historicalData) {
  var prices = historicalData.prices.map(function(p) { return p[1]; });
  var volumes = historicalData.total_volumes.map(function(v) { return v[1]; });
  if (prices.length < 21) return null;

  var n = prices.length;
  var latest = n - 1;
  var dailyReturn = (prices[latest] - prices[latest - 1]) / prices[latest - 1];

  var sum7 = 0;
  for (var i = latest - 6; i <= latest; i++) sum7 += prices[i];
  var ma7 = sum7 / 7;

  var sum14 = 0;
  for (var i = latest - 13; i <= latest; i++) sum14 += prices[i];
  var ma14 = sum14 / 14;

  var priceToMa7 = prices[latest] / ma7;
  var priceToMa14 = prices[latest] / ma14;
  var ma7ToMa14 = ma7 / ma14;

  var returns7 = [];
  for (var i = latest - 6; i <= latest; i++) {
    returns7.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  var mean7r = 0;
  for (var i = 0; i < returns7.length; i++) mean7r += returns7[i];
  mean7r /= returns7.length;
  var variance7 = 0;
  for (var i = 0; i < returns7.length; i++) variance7 += Math.pow(returns7[i] - mean7r, 2);
  var volatility7d = Math.sqrt(variance7 / returns7.length);

  var returns14 = [];
  for (var i = latest - 13; i <= latest; i++) {
    returns14.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  var mean14r = 0;
  for (var i = 0; i < returns14.length; i++) mean14r += returns14[i];
  mean14r /= returns14.length;
  var variance14 = 0;
  for (var i = 0; i < returns14.length; i++) variance14 += Math.pow(returns14[i] - mean14r, 2);
  var volatility14d = Math.sqrt(variance14 / returns14.length);

  var momentum7d = (prices[latest] - prices[latest - 7]) / prices[latest - 7];
  var momentum14d = (prices[latest] - prices[latest - 14]) / prices[latest - 14];

  var gains = 0, losses = 0;
  for (var i = latest - 13; i <= latest; i++) {
    var diff = prices[i] - prices[i - 1];
    if (diff > 0) gains += diff;
    else losses -= diff;
  }
  var avgGain = gains / 14;
  var avgLoss = losses / 14;
  var rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  var rsi14 = 100 - (100 / (1 + rs));

  var volLatest = volumes[latest] || 1;
  var volPrev = volumes[latest - 1] || 1;
  var volumeChange = (volLatest - volPrev) / volPrev;
  var volSum7 = 0;
  for (var i = latest - 6; i <= latest; i++) volSum7 += (volumes[i] || 0);
  var volMa7 = volSum7 / 7;
  var volumeRatio = volMa7 > 0 ? volLatest / volMa7 : 1;

  return {
    daily_return: dailyReturn, price_to_ma7: priceToMa7, price_to_ma14: priceToMa14,
    ma7_to_ma14: ma7ToMa14, volatility_7d: volatility7d, volatility_14d: volatility14d,
    momentum_7d: momentum7d, momentum_14d: momentum14d, rsi_14: rsi14,
    volume_change: volumeChange, volume_ratio: volumeRatio,
  };
}

// == Client-Side Inference ==
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function predict(features, tokenModel, featureNames) {
  var scaled = [];
  for (var i = 0; i < featureNames.length; i++) {
    var f = featureNames[i];
    var val = features[f] != null ? features[f] : 0;
    var mean = tokenModel.scaler_mean[i];
    var std = tokenModel.scaler_std[i];
    scaled.push(std === 0 ? 0 : (val - mean) / std);
  }

  var z = tokenModel.bias;
  for (var i = 0; i < scaled.length; i++) z += tokenModel.weights[i] * scaled[i];

  var probability = sigmoid(z);
  var direction = probability >= 0.5 ? "UP" : "DOWN";
  var confidence = direction === "UP" ? probability : 1 - probability;

  var contributions = [];
  for (var i = 0; i < featureNames.length; i++) {
    contributions.push({
      feature: featureNames[i],
      weight: tokenModel.weights[i],
      scaledValue: scaled[i],
      contribution: tokenModel.weights[i] * scaled[i],
      rawValue: features[featureNames[i]],
    });
  }
  contributions.sort(function(a, b) { return Math.abs(b.contribution) - Math.abs(a.contribution); });

  return { direction: direction, confidence: confidence, probability: probability, contributions: contributions };
}

function getSignalStrength(confidence, direction) {
  if (confidence >= 0.75) return direction === "UP" ? "Strong Buy" : "Strong Sell";
  if (confidence >= 0.60) return direction === "UP" ? "Buy" : "Sell";
  return "Neutral";
}

function getSignalClass(confidence, direction) {
  if (confidence >= 0.75) return direction === "UP" ? "strong-buy" : "strong-sell";
  if (confidence >= 0.60) return direction === "UP" ? "buy" : "sell";
  return "neutral";
}

// == Prediction Engine ==
function buildPredictionEntry(features, tokenModel, featureNames, featureLabels, liveData) {
  var result = predict(features, tokenModel, featureNames);
  var reasons = result.contributions.slice(0, 5).map(function(c) {
    var label = featureLabels[c.feature] || c.feature;
    var isBullish = c.contribution > 0;
    return {
      label: label,
      impact: isBullish ? "Bullish" : "Bearish",
      icon: isBullish ? "\uD83D\uDCC8" : "\uD83D\uDCC9",
      value: c.rawValue,
      strength: Math.abs(c.contribution),
    };
  });
  return {
    direction: result.direction, confidence: result.confidence,
    probability: result.probability, contributions: result.contributions,
    reasons: reasons,
    signal: getSignalStrength(result.confidence, result.direction),
    signalClass: getSignalClass(result.confidence, result.direction),
    accuracy: tokenModel.accuracy,
    trainSamples: tokenModel.train_samples,
    liveData: liveData,
  };
}

async function runPredictions() {
  if (!state.modelData) return;
  var featureNames = state.modelData.features;
  var featureLabels = state.modelData.feature_labels;

  // Phase 1: Instant predictions from pre-computed features
  var keys = Object.keys(TOKEN_MAP);
  for (var k = 0; k < keys.length; k++) {
    var coinId = keys[k];
    var info = TOKEN_MAP[coinId];
    var tokenModel = state.modelData.tokens[info.symbol];
    if (!tokenModel || !tokenModel.latest_features) continue;
    state.predictions[coinId] = buildPredictionEntry(
      tokenModel.latest_features, tokenModel, featureNames, featureLabels, false
    );
  }
  renderPrediction();
  renderComparison();

  // Phase 2: Try to upgrade with live CoinGecko data
  for (var k = 0; k < keys.length; k++) {
    var coinId = keys[k];
    var info = TOKEN_MAP[coinId];
    var tokenModel = state.modelData.tokens[info.symbol];
    if (!tokenModel) continue;
    try {
      var controller = new AbortController();
      var timeout = setTimeout(function() { controller.abort(); }, 8000);
      var url = COINGECKO_API + "/coins/" + coinId + "/market_chart?vs_currency=usd&days=30&interval=daily";
      var resp = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);
      if (!resp.ok) continue;
      var historical = await resp.json();
      var features = computeFeatures(historical);
      if (!features) continue;
      state.predictions[coinId] = buildPredictionEntry(
        features, tokenModel, featureNames, featureLabels, true
      );
      await new Promise(function(r) { setTimeout(r, 1500); });
    } catch (e) {
      // Keep pre-computed prediction
    }
  }
}

// == Load Model Data ==
async function loadModelData() {
  try {
    var resp = await fetch("data/model_data.json");
    if (!resp.ok) throw new Error("Model data: " + resp.status);
    state.modelData = await resp.json();
    console.log("Model loaded:", Object.keys(state.modelData.tokens));
    return true;
  } catch (e) {
    console.warn("No model data found", e);
    return false;
  }
}

// == UI Rendering ==
function renderTokenBar() {
  var bar = document.getElementById("tokenBar");
  bar.innerHTML = "";
  var keys = Object.keys(TOKEN_MAP);
  for (var k = 0; k < keys.length; k++) {
    var coinId = keys[k];
    var info = TOKEN_MAP[coinId];
    var market = state.marketData[coinId];
    var change = market ? market.price_change_percentage_24h : null;
    var isActive = coinId === state.selectedToken;
    var chip = document.createElement("button");
    chip.className = "token-chip" + (isActive ? " active" : "");
    chip.setAttribute("data-coin", coinId);
    chip.onclick = (function(id) { return function() { selectToken(id); }; })(coinId);
    var imgSrc = market ? market.image : "";
    var changeHtml = "";
    if (change != null) {
      changeHtml = '<span class="token-chip__change ' + (change >= 0 ? "up" : "down") + '">' + formatPct(change) + '</span>';
    }
    chip.innerHTML = '<img class="token-chip__icon" src="' + imgSrc + '" alt="' + info.symbol + '" onerror="this.style.display=\'none\'">' +
      '<span>' + info.symbol + '</span>' + changeHtml;
    bar.appendChild(chip);
  }
}

function renderPriceHero() {
  var market = state.marketData[state.selectedToken];
  var info = TOKEN_MAP[state.selectedToken];
  if (!market) return;
  document.getElementById("heroTokenName").textContent = info.name + " (" + info.symbol + ")";
  document.getElementById("heroPrice").textContent = formatPrice(market.current_price);
  var change = market.price_change_percentage_24h;
  var changeEl = document.getElementById("heroChange");
  changeEl.textContent = formatPct(change);
  changeEl.className = "price-hero__change " + (change >= 0 ? "up" : "down");
  document.getElementById("heroVolume").textContent = formatCompact(market.total_volume);
  document.getElementById("heroMcap").textContent = formatCompact(market.market_cap);
  document.getElementById("heroHigh").textContent = formatPrice(market.high_24h);
  document.getElementById("heroLow").textContent = formatPrice(market.low_24h);
  var rank = market.market_cap_rank;
  var rankEl = document.getElementById("heroRank");
  rankEl.textContent = rank ? "#" + rank : "";
  rankEl.style.background = "var(--accent-glow)";
  rankEl.style.color = "var(--accent-primary)";
}

function renderPrediction() {
  var pred = state.predictions[state.selectedToken];
  var badge = document.getElementById("predBadge");
  var arrow = document.getElementById("predArrow");
  var dirEl = document.getElementById("predDirection");
  var confValue = document.getElementById("predConfidence");
  var confBar = document.getElementById("predConfBar");
  var signalEl = document.getElementById("predSignal");
  var accuracyEl = document.getElementById("predAccuracy");
  var samplesEl = document.getElementById("predSamples");
  var explainList = document.getElementById("explainList");

  if (!pred) {
    badge.className = "prediction__badge";
    badge.style.background = "rgba(255,255,255,0.03)";
    badge.style.border = "1px solid var(--border)";
    badge.style.boxShadow = "none";
    arrow.textContent = "\u23F3";
    dirEl.textContent = "Awaiting Model";
    confValue.textContent = "\u2014";
    confBar.style.width = "0%";
    signalEl.textContent = "No model loaded";
    signalEl.className = "signal__text neutral";
    accuracyEl.textContent = "\u2014";
    samplesEl.textContent = "";
    explainList.innerHTML = '<div class="error-message"><div class="error-message__text">Run the ML pipeline first</div><div class="error-message__sub">python ml/train_model.py</div></div>';
    return;
  }

  var isUp = pred.direction === "UP";
  badge.className = "prediction__badge " + (isUp ? "up" : "down");
  badge.style.background = "";
  badge.style.border = "";
  badge.style.boxShadow = "";
  arrow.textContent = isUp ? "\u25B2" : "\u25BC";
  dirEl.textContent = pred.direction;
  var confPct = (pred.confidence * 100).toFixed(1) + "%";
  confValue.textContent = confPct;
  confBar.className = "confidence__fill " + (isUp ? "up" : "down");
  confBar.style.width = confPct;
  signalEl.textContent = pred.signal;
  signalEl.className = "signal__text " + pred.signalClass;
  accuracyEl.textContent = (pred.accuracy * 100).toFixed(1) + "%";
  samplesEl.textContent = pred.trainSamples + " training samples";

  var html = "";
  for (var i = 0; i < pred.reasons.length; i++) {
    var r = pred.reasons[i];
    html += '<div class="explain__item">' +
      '<span class="explain__icon">' + r.icon + '</span>' +
      '<span class="explain__text">' + r.label + '</span>' +
      '<span class="explain__impact ' + (r.impact === "Bullish" ? "bullish" : "bearish") + '">' + r.impact + '</span>' +
      '</div>';
  }
  explainList.innerHTML = html;
}

function renderChart() {
  var chartData = state.chartData[state.selectedToken];
  if (!chartData) return;
  var prices = chartData.prices;
  var labels = [];
  var values = [];
  for (var i = 0; i < prices.length; i++) {
    var d = new Date(prices[i][0]);
    if (state.chartDays <= 7) {
      labels.push(d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" }));
    } else {
      labels.push(d.toLocaleDateString("en-US", { month: "short", day: "numeric" }));
    }
    values.push(prices[i][1]);
  }

  var ctx = document.getElementById("priceChart").getContext("2d");
  var isUp = values[values.length - 1] >= values[0];
  var gradient = ctx.createLinearGradient(0, 0, 0, 320);
  if (isUp) {
    gradient.addColorStop(0, "rgba(16, 185, 129, 0.25)");
    gradient.addColorStop(1, "rgba(16, 185, 129, 0)");
  } else {
    gradient.addColorStop(0, "rgba(239, 68, 68, 0.25)");
    gradient.addColorStop(1, "rgba(239, 68, 68, 0)");
  }

  if (state.priceChart) state.priceChart.destroy();
  state.priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        data: values,
        borderColor: isUp ? "#10b981" : "#ef4444",
        backgroundColor: gradient,
        borderWidth: 2, fill: true, tension: 0.4,
        pointRadius: 0, pointHoverRadius: 5,
        pointHoverBackgroundColor: isUp ? "#10b981" : "#ef4444",
        pointHoverBorderColor: "#fff", pointHoverBorderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(17, 24, 39, 0.95)",
          titleColor: "#f1f5f9", bodyColor: "#94a3b8",
          borderColor: "rgba(148, 163, 184, 0.1)", borderWidth: 1,
          cornerRadius: 8, padding: 12, displayColors: false,
          callbacks: {
            title: function(items) { return items[0].label; },
            label: function(item) { return formatPrice(item.raw); },
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(148, 163, 184, 0.05)" },
          ticks: { color: "#64748b", font: { size: 11 }, maxTicksLimit: 8, maxRotation: 0 },
          border: { display: false },
        },
        y: {
          grid: { color: "rgba(148, 163, 184, 0.05)" },
          ticks: { color: "#64748b", font: { size: 11 }, callback: function(v) { return formatPrice(v); } },
          border: { display: false },
        },
      },
    },
  });
}

function renderComparison() {
  var body = document.getElementById("comparisonBody");
  var entries = [];
  var keys = Object.keys(TOKEN_MAP);
  for (var k = 0; k < keys.length; k++) {
    var coinId = keys[k];
    var info = TOKEN_MAP[coinId];
    var market = state.marketData[coinId];
    var pred = state.predictions[coinId];
    var score = pred ? (pred.direction === "UP" ? pred.confidence : -pred.confidence) : 0;
    entries.push({ coinId: coinId, info: info, market: market, pred: pred, score: score });
  }
  entries.sort(function(a, b) { return b.score - a.score; });

  var html = "";
  for (var idx = 0; idx < entries.length; idx++) {
    var e = entries[idx];
    var coinId = e.coinId;
    var info = e.info;
    var market = e.market;
    var pred = e.pred;
    var change = market ? market.price_change_percentage_24h : null;
    var predHtml = pred
      ? '<span class="table__prediction ' + (pred.direction === "UP" ? "up" : "down") + '">' + (pred.direction === "UP" ? "\u25B2" : "\u25BC") + " " + pred.direction + '</span>'
      : '<span style="color: var(--text-muted)">\u2014</span>';

    html += '<tr onclick="selectToken(\'' + coinId + '\')">' +
      '<td style="color: var(--text-muted); font-weight: 600;">' + (idx + 1) + '</td>' +
      '<td><div class="table__token">' +
        '<img class="table__token-icon" src="' + (market ? market.image : "") + '" alt="' + info.symbol + '" onerror="this.style.display=\'none\'">' +
        '<div><div class="table__token-name">' + info.name + '</div><div class="table__token-symbol">' + info.symbol + '</div></div>' +
      '</div></td>' +
      '<td class="table__price">' + (market ? formatPrice(market.current_price) : "\u2014") + '</td>' +
      '<td class="table__change ' + (change >= 0 ? "up" : "down") + '">' + (change != null ? formatPct(change) : "\u2014") + '</td>' +
      '<td>' + predHtml + '</td>' +
      '<td class="table__confidence">' + (pred ? (pred.confidence * 100).toFixed(1) + "%" : "\u2014") + '</td>' +
      '<td class="table__signal ' + (pred ? pred.signalClass : "") + '">' + (pred ? pred.signal : "\u2014") + '</td>' +
      '<td><canvas class="table__sparkline" id="spark-' + coinId + '" width="100" height="32"></canvas></td>' +
      '</tr>';
  }
  body.innerHTML = html;

  requestAnimationFrame(function() {
    for (var idx = 0; idx < entries.length; idx++) {
      var e = entries[idx];
      var canvas = document.getElementById("spark-" + e.coinId);
      if (!canvas || !e.market) continue;
      var sparkline = e.market.sparkline_in_7d ? e.market.sparkline_in_7d.price : [];
      if (sparkline.length < 2) continue;
      renderSparkline(canvas, sparkline);
    }
  });
}

function renderSparkline(canvas, data) {
  var ctx = canvas.getContext("2d");
  var w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  var min = data[0], max = data[0];
  for (var i = 1; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  var range = max - min || 1;
  var isUp = data[data.length - 1] >= data[0];
  ctx.beginPath();
  ctx.strokeStyle = isUp ? "#10b981" : "#ef4444";
  ctx.lineWidth = 1.5;
  for (var i = 0; i < data.length; i++) {
    var x = (i / (data.length - 1)) * w;
    var y = h - ((data[i] - min) / range) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// == Token Selection ==
async function selectToken(coinId) {
  if (!TOKEN_MAP[coinId]) return;
  state.selectedToken = coinId;
  renderTokenBar();
  renderPriceHero();
  renderPrediction();
  try {
    await fetchChartData(coinId, state.chartDays);
    renderChart();
  } catch (e) {
    console.warn("Chart fetch failed:", e);
  }
}
window.selectToken = selectToken;

// == App Controller ==
var app = {
  init: async function() {
    console.log("TokenTrend initializing...");
    await loadModelData();

    try {
      await fetchMarketData();
      showToast("Live data loaded from CoinGecko");
    } catch (e) {
      console.error("Market data failed:", e);
      showToast("CoinGecko unavailable - showing predictions only");
      setTimeout(function() {
        fetchMarketData().then(function() {
          renderTokenBar(); renderPriceHero(); renderComparison();
        }).catch(function() {});
      }, 60000);
    }

    renderTokenBar();
    renderPriceHero();
    renderPrediction();
    renderComparison();

    try {
      await fetchChartData(state.selectedToken, state.chartDays);
      renderChart();
    } catch (e) {
      console.warn("Initial chart failed:", e);
    }

    if (state.modelData) {
      showToast("Running AI predictions...");
      await runPredictions();
      renderPrediction();
      renderComparison();
      showToast("AI predictions complete");
    }

    document.getElementById("lastUpdate").textContent =
      "Updated " + new Date().toLocaleTimeString();
    var self = this;
    setInterval(function() { self.refresh(); }, 120000);
  },

  refresh: async function() {
    var btn = document.getElementById("refreshBtn");
    btn.classList.add("loading");
    try {
      await fetchMarketData();
      renderTokenBar(); renderPriceHero(); renderComparison();
      await fetchChartData(state.selectedToken, state.chartDays);
      renderChart();
      if (state.modelData) {
        await runPredictions();
        renderPrediction(); renderComparison();
      }
      document.getElementById("lastUpdate").textContent =
        "Updated " + new Date().toLocaleTimeString();
      showToast("Data refreshed");
    } catch (e) {
      showToast("Refresh failed - API may be rate limited");
    }
    btn.classList.remove("loading");
  },

  setChartRange: async function(days, btn) {
    state.chartDays = days;
    var tabs = document.querySelectorAll(".chart-tab");
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.remove("active");
    btn.classList.add("active");
    try {
      await fetchChartData(state.selectedToken, days);
      renderChart();
    } catch (e) {
      showToast("Chart fetch failed");
    }
  },
};

window.app = app;
document.addEventListener("DOMContentLoaded", function() { app.init(); });
