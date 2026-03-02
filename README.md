# Options Flow Predictor

## Overview

The Options Flow Predictor is a machine learning system that forecasts short-term stock price movements by analyzing institutional options trading patterns. The model identifies when large traders are positioning for specific directional moves by examining unusual options activity, dealer positioning, and market sentiment indicators.

## How It Works

### Core Theory

Large institutional traders often reveal their market expectations through options trading before making moves in the underlying stock. When hedge funds, pension funds, or sophisticated traders expect a stock to move, they frequently establish options positions first due to leverage advantages and risk management benefits.

The model captures these "smart money" signals by monitoring:

**Put/Call Ratio Analysis**
- When institutional traders expect upward moves, call buying increases relative to put buying
- Heavy put buying often precedes downward price movements
- The model tracks volume-weighted put/call ratios to identify sentiment shifts

**Unusual Volume Detection**
- Compares current options volume to historical patterns
- Identifies when trading activity exceeds normal levels by significant margins
- High volume often precedes major price movements as institutions build positions

**Dealer Positioning Analysis**
- Options dealers (market makers) hedge their exposures by trading the underlying stock
- When dealers are net long gamma, they buy stocks as prices fall and sell as prices rise
- This creates support and resistance levels that the model identifies and exploits

**Volatility Structure Patterns**
- Examines implied volatility across different strike prices and expiration dates
- Identifies when institutions are paying up for specific directional exposure
- Detects volatility skew patterns that indicate expected price movements

### Machine Learning Approach

The system employs ensemble learning combining multiple algorithms:

**Random Forest Component**
- Captures non-linear relationships between options flow features and price movements
- Handles high-dimensional options data effectively
- Provides feature importance rankings to identify key predictive signals

**XGBoost Component**
- Optimized for sequential pattern recognition in time series data
- Handles missing data and outliers common in options markets
- Provides gradient-boosted predictions for enhanced accuracy

**Ensemble Integration**
- Combines predictions from multiple models to reduce overfitting
- Weights models based on historical performance
- Provides more robust signals than individual algorithms

### Feature Engineering

The model transforms raw options data into predictive features:

**Technical Indicators**
- Momentum indicators like RSI to capture price trend information
- Volatility forecasts using statistical models
- Volume patterns and trading intensity measures

**Options-Specific Features**
- Gamma exposure calculations showing dealer hedging pressures
- At-the-money implied volatility levels
- Risk reversal patterns indicating directional bias

**Market Regime Detection**
- Identifies different market conditions (low volatility, high volatility, trending, ranging)
- Adjusts predictions based on current market regime
- Incorporates economic indicators for broader context

## Model Outputs

**Price Movement Predictions**
- Forecasts 1-day, 3-day, and 5-day percentage price changes
- Provides probability estimates for directional moves
- Indicates expected magnitude of movements

**Trading Signals**
- Clear buy/sell/hold recommendations based on prediction confidence
- Risk assessment for each signal
- Position sizing suggestions based on signal strength

**Market Intelligence**
- Unusual activity alerts for specific symbols
- Dealer positioning summaries
- Volatility regime classifications

## Target Performance

The model aims to achieve:
- Daily returns exceeding 40 basis points on average
- Sharpe ratios above 2.0 through consistent performance
- High hit rates on directional predictions
- Minimal drawdown periods during adverse conditions

## Methodology Advantages

**Real-Time Analysis**
- Processes current options flow data for immediate insights
- Updates predictions as new trading information becomes available
- Provides actionable signals during market hours

**Multi-Asset Coverage**
- Analyzes major ETFs representing different market segments
- Identifies sector rotation patterns through relative signal strength
- Enables portfolio-level strategy implementation

**Risk-Aware Predictions**
- Incorporates volatility forecasts into position recommendations
- Provides confidence intervals around predictions
- Flags high-risk periods for reduced position sizing

The Options Flow Predictor transforms complex institutional trading patterns into actionable investment signals, providing retail and institutional traders with insights typically available only to market makers and sophisticated hedge funds.

## MVP Live Grátis (Polling)

Este repositório inclui o script `live_options_polling.py` para rodar um monitoramento *near-real-time* com custo zero inicial:

- Yahoo/yfinance para preço + cadeia de opções
- VIX e VIX9D para contexto de volatilidade
- Recalcula sinais a cada 5 minutos (`--poll-seconds 300`)
- Apenas imprime sinais no terminal (sem webhook/email)

### Executar

```bash
python live_options_polling.py --symbols SPY,QQQ,IWM --poll-seconds 300
```

Rodar um único ciclo (teste):

```bash
python live_options_polling.py --symbols SPY,QQQ,IWM --once
```

Validar dependências e coleta (self-check):

```bash
python live_options_polling.py --symbols SPY,QQQ,IWM --self-check
```

> Observação: o script usa dados gratuitos e polling, então não substitui feed profissional tick-by-tick.

### Fidelidade em relação ao notebook

Este script é **parcialmente fiel** ao notebook:
- **Fiel** nas regras de sinal de fluxo de opções: `pcr_volume` (bullish < 0.7, bearish > 1.0) e `uoa_ratio > 1.25` para volume incomum.
- **Não fiel** à parte de ML/ensemble e engenharia completa de features do notebook (Random Forest, XGBoost, treino, target de 1/3/5 dias, etc.).
- **Não fiel** ao bloco de interpretação/risk completo; aqui o foco é monitoramento simples em tempo quase real por polling.

## Sinais de entrada com ML (mais próximo do notebook)

Para sair do modo puramente rule-based e aproximar do notebook:

1. Treine e salve um modelo com features históricas:

```bash
python train_live_model.py --symbols SPY,QQQ,IWM --out artifacts/live_model.pkl
```

2. Rode inferência live combinando ML + fluxo de opções:

```bash
python live_ml_signals.py --model artifacts/live_model.pkl --symbols SPY,QQQ,IWM --poll-seconds 300
```

Regra de ação no script live:
- `LONG`: ML bullish **alinhado** com `pcr_signal=bullish` e magnitude mínima
- `SHORT`: ML bearish **alinhado** com `pcr_signal=bearish` e magnitude mínima
- `NO_TRADE`: conflito/força fraca

> Nota: isso é uma aproximação operacional do notebook; para fidelidade máxima, migrar toda engenharia de features do notebook para módulo Python reutilizável.
