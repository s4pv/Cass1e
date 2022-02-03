# Cassandra Project
1. pip install -r requirements.txt
2. First create keys.py with public (Pkey) and secret key (Skey) info of your binance account to connect to the API.
Once connected to the API run data.py to extract databases (csv) to train forecasting models.
Example: Daily Ethereum OHLC and volume data.
3. Choose best fit between legacy finance models and machine learning models according to asset data.
Statistical tests and plots if wanted.
4. With the chosen model, fit and forecast the assets in tickers and create model limits to trade.
5. With the forecast, the bot can trade and optimize a portfolio.

#Pending Modules
1. Optimization modules for the following models:
   1. Machine Learning (only new models):
      1. Grid Search Framework
      2. Hyperparameters
   2. Finance (only few models):
      1. Grid Search Framework
2. Comparing/Scoring(Cross-Validation)/Statistical tests for different models and select best fit model (maybe stats module not needed)
3. Model Forecast (Separate Fit from forecast.)
4. Plots for: stats, finance, machinelearning, modelforecast and portfolio (ok but move from portfolio to plots)
5. Multi-outputs (OHLC) for forecasting
6. BOT:
   1. Trading Module based on model limits
   2. Portfolio Managing and MTP optimization (ok)
   3. Bot loop

#Further investigation
1. Hybrid models (ex: ARIMA + LSTM)
2. Interface
3. Telegram API