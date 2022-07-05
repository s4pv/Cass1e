# Cassandra Project
Cassandra aims to:
1. Forecast Finance models
2. Manage a portfolio
3. Trade over that portfolio and forecast

# Requirements and functionality
1. pip install -r requirements.txt
2. First create keys.py with public (Pkey) and secret key (Skey) info of your binance account to connect to the API.
Once connected to the API run data.py to extract databases (csv) to train forecasting models.
Example: Daily Ethereum OHLC and volume data.
3. Model: Non-bidirectional multivariable (OHLV) multi-step LSTM to predict (C). Statistical tests and plots
if required/wanted.
4. With the chosen model, fit and forecast the assets in tickers and create finance model boundaries.
5. With the forecast, the bot can trade and optimize portfolios.

# Further steps
1. Update data on already fitted model
2. frac-diff, Grubbs
3. CLI  or interface Telegram API
4. plots with confidence intervals