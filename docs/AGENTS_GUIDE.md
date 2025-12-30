# AI Market Analysis System - Agent Guide

## Overview

The system uses **10 specialized AI agents** that work together to analyze markets and generate trading signals. Each agent focuses on a specific aspect of market analysis.

---

## 🤖 Individual Agents

### 1. MomentumAgent
**Purpose:** Detects price trends and momentum

**How it works:**
- Analyzes price velocity and acceleration
- Uses technical indicators: RSI, MACD, Bollinger Bands
- Identifies trend strength and direction
- Signals: BUY on strong upward momentum, SELL on downward momentum

**Best for:** Trending markets, breakout detection

---

### 2. SentimentAgent
**Purpose:** Analyzes market sentiment from news and social media

**How it works:**
- Processes news articles via RAG system
- Analyzes social media sentiment
- Uses NLP to determine bullish/bearish sentiment
- Weights recent news more heavily

**Best for:** News-driven moves, earnings reactions

---

### 3. RiskAgent
**Purpose:** Assesses portfolio and position risk

**How it works:**
- Calculates Value at Risk (VaR)
- Monitors portfolio volatility
- Tracks correlation between positions
- Identifies concentration risk

**Best for:** Position sizing, risk management

---

### 4. StrategyAgent
**Purpose:** Develops swing trading strategies

**How it works:**
- Identifies support/resistance levels
- Calculates optimal entry/exit points
- Sets stop-loss and take-profit levels
- Considers multi-day holding periods

**Best for:** Swing trading (3-10 day holds)

---

### 5. TechnicalAgent
**Purpose:** Pure technical analysis

**How it works:**
- Chart pattern recognition
- Candlestick analysis
- Moving average crossovers
- Volume analysis

**Best for:** Day trading, technical setups

---

### 6. FundamentalAgent
**Purpose:** Analyzes company fundamentals

**How it works:**
- Evaluates P/E, P/B, revenue growth
- Assesses balance sheet strength
- Monitors earnings trends
- Compares to sector peers

**Best for:** Long-term positions, value investing

---

### 7. VolatilityAgent
**Purpose:** Tracks and predicts volatility

**How it works:**
- Calculates implied vs realized volatility
- Monitors VIX correlation
- Predicts volatility expansion/contraction
- Identifies volatility breakouts

**Best for:** Options trading, risk timing

---

### 8. MacroAgent
**Purpose:** Analyzes macroeconomic factors

**How it works:**
- Tracks interest rates and Fed policy
- Monitors economic indicators (GDP, unemployment)
- Analyzes sector rotation
- Considers global market correlations

**Best for:** Sector allocation, macro positioning

---

### 9. PatternAgent
**Purpose:** Identifies chart patterns

**How it works:**
- Detects head & shoulders, triangles, flags
- Measures pattern completion probability
- Calculates target prices from patterns
- Tracks historical pattern success rates

**Best for:** Breakout trading, pattern-based entries

---

### 10. MLPredictorAgent
**Purpose:** Machine learning price predictions

**How it works:**
- Uses LSTM neural networks for price forecasting
- Trains on historical price data
- Incorporates multiple timeframes
- Continuously retrains on new data

**Best for:** Price target predictions, trend direction

---

## 🎯 Meta Agents (Combine Individual Agents)

### EnsembleBlender
**Purpose:** Combines all agent signals into one recommendation

**How it works:**
```
1. Collects signals from all 10 agents
2. Weights agents by recent performance
3. Considers market regime (bull/bear/sideways)
4. Produces final blended signal with confidence score
```

**Weighting Example:**
| Market Regime | Top Weighted Agents |
|---------------|---------------------|
| Bull Market   | Momentum, Technical |
| Bear Market   | Risk, Volatility    |
| Sideways      | Pattern, Strategy   |

---

### Meta-Evaluation Agent
**Purpose:** Ranks agent performance and adjusts weights

**How it works:**
- Tracks each agent's prediction accuracy
- Calculates Sharpe ratio for each agent
- Adjusts ensemble weights based on performance
- Demotes underperforming agents

---

### RL Strategy Agent
**Purpose:** Reinforcement learning for optimal actions

**How it works:**
- Learns from historical trade outcomes
- Optimizes for risk-adjusted returns
- Adapts to changing market conditions
- Uses PPO algorithm for policy updates

---

### RAG Event Agent
**Purpose:** Retrieval-Augmented Generation for news analysis

**How it works:**
- Indexes news articles in vector database
- Retrieves relevant context for queries
- Uses Ollama LLM (llama3.1) for analysis
- Provides real-time news impact assessment

---

## 📊 Signal Interpretation

| Confidence | Meaning | Action |
|------------|---------|--------|
| 80%+ | Very Strong | Consider large position |
| 70-80% | Strong | Normal position size |
| 60-70% | Moderate | Smaller position |
| 50-60% | Weak | Caution / Hold |
| <50% | No Signal | Wait |

---

## 🔄 How Agents Work Together

```
Market Data → Individual Agents (10) → Signals
                     ↓
              EnsembleBlender → Blended Signal
                     ↓
              Meta-Evaluation → Performance Tracking
                     ↓
              RL Strategy → Optimal Action
                     ↓
              Final Trading Decision
```

---

## 🛠️ API Endpoints for Agent Data

| Endpoint | Description |
|----------|-------------|
| `/agents/performance` | Agent performance metrics |
| `/ensemble-blender/summary` | Blended signal summary |
| `/meta-evaluation/agent-rankings` | Agent rankings by regime |
| `/rl-strategy/recommendation` | RL-based recommendation |
| `/rag-event-agent/summary` | News analysis summary |

---

## 💡 Best Practices

1. **Trust the Ensemble** - The blended signal considers all factors
2. **Check Confidence** - Only act on >70% confidence signals
3. **Consider Regime** - Bull/bear market affects agent accuracy
4. **Monitor Rankings** - Top-ranked agents are more reliable
5. **Use Stop Losses** - Even high-confidence signals can fail
