# DQN Algorithmic Trading & Portfolio Optimizer
**CS5100 — Foundations of Artificial Intelligence**
Nathan Chen

A Deep Q-Network (DQN) reinforcement learning agent that learns when to buy and sell stocks using technical indicators and market data. Includes a portfolio optimizer that picks the best combination of stocks and allocates capital across them.


## Setup

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```


## How to Run

**Backtest on default tickers (AAPL, MSFT, TSLA):**
```
python3 stock_prediction/main.py
```

**Backtest a specific ticker:**
```
python3 stock_prediction/main.py --ticker NVDA
```

**Get a live buy/sell signal for today:**
```
python3 stock_prediction/main.py --signal --ticker AAPL
```

**Build and optimize a portfolio:**
```
python3 stock_prediction/main.py --portfolio
```

**Run stress tests (2008, 2020, 2022 crashes):**
```
python3 stock_prediction/main.py --stress-test
```

**Compare DQN vs rule-based strategies (MA crossover, RSI, momentum):**
```
python3 stock_prediction/main.py --benchmark
python3 stock_prediction/main.py --benchmark --ticker NVDA JPM NKE
```

**Walk-forward validation across 3 time periods:**
```
python3 stock_prediction/main.py --walk-forward
python3 stock_prediction/main.py --walk-forward --walk-forward-episodes 300
```

**Test on the broad 30+ ticker universe across multiple sectors:**
```
python3 stock_prediction/main.py --broad
python3 stock_prediction/main.py --broad --benchmark
python3 stock_prediction/main.py --broad --walk-forward
```

Results (equity curves, confusion matrices, charts) are saved to the `results/` folder.

## Live Signal Setup

Create a `.env` file in the project root with the following:

```
# Email alerts (optional)
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
ALERT_TO=your_gmail@gmail.com

# Alpaca trading (optional)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```

**Gmail app password:** Go to myaccount.google.com/security → enable 2-Step Verification → search "App passwords" → create one named "Trading Monitor" → paste the 16-character password into SMTP_PASSWORD.

**Alpaca API keys:** Sign up at alpaca.markets → go to Paper Trading → API Keys → generate a key pair. Set `ALPACA_PAPER=false` only when you are ready to trade real money.


## Trading Commands

**Paper trade (simulated money, real signals):**
```
python3 stock_prediction/main.py --monitor --alpaca
```

**Paper trade with email alerts:**
```
python3 stock_prediction/main.py --monitor --alpaca --email
```

**Live trade (real money — be careful):**
```
python3 stock_prediction/main.py --monitor --alpaca --live
```




## Project Structure

```
FINAL PROJECT/
├── stock_prediction/
│   ├── main.py            — entry point, CLI flags
│   ├── rl_agent.py        — DQN agent and trading environment
│   ├── features.py        — technical indicator feature engineering
│   ├── portfolio.py       — portfolio screening and Markowitz optimization
│   ├── backtesting.py     — equity curve, Sharpe ratio, drawdown metrics
│   ├── evaluation.py      — classification metrics and confusion matrix
│   ├── models.py          — data splitting and feature scaling
│   ├── data_collection.py — downloads stock data via yfinance
│   ├── benchmarks.py      — rule-based strategies + DQN comparison
│   ├── walk_forward.py    — rolling window validation across time periods
│   ├── live_signal.py     — generates a signal for today
│   ├── stress_test.py     — bear market evaluation
│   ├── alpaca_trader.py   — Alpaca API order execution
│   └── monitor.py         — continuous live monitoring
├── results/               — output charts (auto-created)
└── requirements.txt
```

---

## How It Works

The DQN agent is trained on 10 years of daily price data (2015–2025). It observes 31 features per day, technical indicators like RSI, MACD, Bollinger Bands, moving averages, ATR, volume, and SPY market context, and learns to choose between buying/holding or moving to cash.

The trading environment enforces real rules: a minimum 2:1 reward-to-risk ratio, ATR-based stop losses, volume confirmation, and a 10-day maximum hold period. This teaches the agent discipline rather than just pattern matching.

The portfolio builder trains a separate DQN for each candidate stock, selects the top performers by Sharpe ratio, then uses Markowitz mean-variance optimization to find the capital allocation that maximizes the combined portfolio Sharpe ratio.
