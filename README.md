# Cassandra Project
Cassandra aims to:
1. Forecast Finance models
2. Manage a portfolio
3. Trade over that forecast and portfolio

# Requirements and functionality today
1. pip install -r requirements.txt
2. First create keys.py with public (Pkey) and secret key (Skey) info of your binance account to connect to the API.
Once connected to the API run data.py to extract databases (csv) to train forecasting models.
Example: Daily BTC OHLC and volume data.
3. Data Engineering: Fractional Differentiation
4. Model: Non-bidirectional multivariable (OHLV) multi-step LSTM to predict (C). Statistical tests and plots
if required/wanted.
4. With the chosen model, fit and forecast the assets in tickers and create finance model boundaries.


# Further steps
1. With the forecast, the bot optimize portfolios and trades over them.
2. Telegram Interface/reporting

