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
3. Manually choose best fit between legacy finance models and machine learning models according to asset data.
Statistical tests and plots if wanted.
4. With the chosen model, fit and forecast the assets in tickers and create finance model boundaries.
5. With the forecast, the bot can trade and optimize portfolios.

#Further investigation
1. Multi-inputs (OHLV) to forecast C
2. Hybrid models (ex: ARIMA + LSTM)
3. CLI
4. Telegram API