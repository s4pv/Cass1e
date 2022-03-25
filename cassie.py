# Welcome To Cassie BoT
# Intended to auto trade some basic strategies on crypto assets. Exchange Binance
# Install requirements
# Add your keys to read and trade your binance account
# Config the bot on the config.py file

from binance.client import Client
from datetime import datetime
import tzlocal
import warnings
import keys
from portfolio import Portfolio
from trades import Trades
from asianrange import AsianRange
from helper import Helper
from datapreparation import Datapreparation
from machinelearning import MachineLearning
from modelforecast import ModelForecast
from tradesignals import Tradesignals
from stats import Stats
import pandas as pd
import numpy

warnings.filterwarnings("ignore")


class Cassie:

    print('----------------------------------')
    print('|       Welcome to Cass1e        |')
    print('----------------------------------')

    # Configuration and class variables
    parsed_config = Helper.load_config('config.yml')

    NO_DATA = parsed_config['model_options']['NO_DATA']

    PAIR_WITH = parsed_config['trading_options']['PAIR_WITH']
    TICKERS_LIST = parsed_config['trading_options']['TICKERS_LIST']
    WORK_TIMEFRAME = parsed_config['trading_options']['WORK_TIMEFRAME']
    TRAIN_TIMEFRAME = parsed_config['trading_options']['TRAIN_TIMEFRAME']

    tickers = [line.strip() for line in open(TICKERS_LIST)]

    # Connecting to the account through the Binance API client
    client = Client(keys.Pkey, keys.Skey)

    # Defining time zones of interest.
    frankfurt_open = 4
    frankfurt_close = 12
    london_open = 5
    london_close = 13
    new_york_open = 10
    new_york_close = 18
    sydney_open = 18
    sydney_close = 2
    tokyo_open = 20
    tokyo_close = 4

    # Defining current timezone and time
    utc_unaware = datetime.now()
    print('We are in ' + str(tzlocal.get_localzone()) + ' timezone.')
    utc_aware = utc_unaware.replace(tzinfo=tzlocal.get_localzone())
    local_aware = datetime(utc_aware.year, utc_aware.month, utc_aware.day, utc_aware.hour, utc_aware.minute)
    print('Current datetime is ' + str(local_aware))

    # Constructor and instance variables
    def __init__(self, name):
        self.name = name


def main():
    cassie = Cassie("Cass1e")
    data_returns = pd.DataFrame()
    for coin in cassie.client.get_all_tickers():
        for crypto in cassie.tickers:
            if crypto + cassie.PAIR_WITH == coin['symbol']:

                # Daily closure: 21 hours. Time to forecast. If monday, also time to fit again the model.
                if True:#str(cassie.local_aware.hour) == 21 and cassie.local_aware.minute == 5:
                    if coin['symbol'] == 'BTCUSDT':
                        if cassie.local_aware.day == 'MONDAY':
                            print('Today is monday. Heaviest day of the week...')
                            print('We have to fit models again to forecast BTCUSDT.')
                            print('Extracting pricing data from the server to model...')
                            dataset = cassie.client.get_klines(symbol=coin['symbol'], interval=cassie.TRAIN_TIMEFRAME,
                                                               limit=cassie.NO_DATA)
                            dataframe = Datapreparation.OHLCV_DataFrame(dataset)
                            #Stats.Shapiro_Wilk(dataframe)
                            print('Fitting the model to the new data')
                            print('This may take a while...')
                            MachineLearning.LSTM(dataframe, coin)
                            print('Model fitted. Results and plots can be seen on respective folders.')
                    print('Extracting pricing data from the server to forecast the coin: ' + coin['symbol'])
                    dataset = cassie.client.get_klines(symbol=coin['symbol'], interval=cassie.WORK_TIMEFRAME,
                                                       limit=cassie.NO_DATA)
                    dataset_ohlcv = Datapreparation.OHLCV_DataFrame(dataset)
                    print('Making forecasts!. Plots can be seen on respective folders.')
                    forecast = ModelForecast.Predict_LSTM(dataset_ohlcv, coin)
                    # with 1000 days of memory for portfolio optimization
                    #ds_forecast = numpy.append(dataset_ohlcv['close'], forecast[0], 0)
                    # without memory
                    ds_forecast = forecast[0]
                    returns = Portfolio.Return(ds_forecast, coin)
                    data_returns[coin['symbol']] = returns[coin['symbol']]
                    #print(data_returns[coin['symbol']])
                    #print(data_returns[coin['symbol']].shape)

                # Forcing Portfolio rebalance and Deleting orders at the end of the new york season. 18 hours localtime.
                if str(cassie.local_aware.hour) == cassie.new_york_close and cassie.local_aware.minute == 0:
                    print('We ended the new york trading hours. Checking for additional portfolio re-balancing.')
                    print('Checking the crypto: ' + coin['symbol'])
                    print('Deleting pending Orders')
                    Trades.Delete_Order(coin['symbol'])
                    Trades.MarketBuy(coin['symbol'])
                    Trades.MarketSell(coin['symbol'])

                # Entering the ending hours of the NY time from 13 to 18 hours localtime. Time to collect the profit!.
                if cassie.london_close < int(cassie.local_aware.hour) <= cassie.new_york_close:
                    print('We entered the ending hours of the NY timezone. Time for take profit!')
                    print('Checking the balance on the crypto: ' + crypto)
                    print('Trying to take profit')
                    #if Portfolio.Optimum_Accounting(assets, optimal_variance_weights, coin['symbol']) < Portfolio.Accounting(coin['symbol']):
                    #    print('Forcing long on the crypto pair:' + coin['symbol'])
                    #    Trades.LimitBuy(coin['symbol'])
                    #if Portfolio.Optimum_Accounting(assets, optimal_variance_weights, coin['symbol']) > Portfolio.Accounting(coin['symbol']):
                    #    print('Sell signal found, trying to go short on the crypto pair:' + coin['symbol'])
                    #    Trades.LimitSell(coin['symbol'])

                # End of the working season, london time: 13 hours. Time to stop working!
                if int(cassie.local_aware.hour) == cassie.london_close and cassie.local_aware.minute == 0:
                    print('We ended the day. That means its time to stop working!')

                # Entering the first hours of the frankfurt/london timezone from 4 to 13 hours. Time to open some.
                if cassie.frankfurt_open < int(cassie.local_aware.hour) < cassie.london_close:
                    print('We entered the trading hours. That means the frankfurt/london timezone.')
                    print('looking forward to rebalance the portfolio.')
                    print('Checking entries for the crypto: ' + coin['symbol'])
                    dataset = cassie.client.get_klines(symbol=coin['symbol'], interval=cassie.WORK_TIMEFRAME,
                                                       limit=cassie.NO_DATA)
                    dataframe = Datapreparation.OHLCV_DataFrame(dataset)
                    if Portfolio.Optimum_Accounting(assets, optimal_variance_weights, coin['symbol']) < Portfolio.Accounting(coin['symbol']):
                        if dataset[0]['price'] < lower_limit:
                            print('Price under the asian range. Looking for long signaling.')
                            buy_sign = Tradesignals.Looking_Long(coin['symbol'], dataframe)
                            if buy_sign == True:
                                print('Bull signal found, trying to go long on the crypto pair:' + coin['symbol'])
                                Trades.LimitBuy(coin['symbol'])
                                buy_sign = False
                    if Portfolio.Optimum_Accounting(assets, optimal_variance_weights, coin['symbol']) > Portfolio.Accounting(coin['symbol']):
                        if dataset[0]['price'] > upper_limit:
                            print('Price above the asian range. Looking for short signaling.')
                            sell_sign = Tradesignals.Looking_Short(coin['symbol'], dataframe)
                            if sell_sign == True:
                                print('Sell signal found, trying to go short on the crypto pair:' + coin['symbol'])
                                Trades.LimitSell(coin['symbol'])
                                sell_sign = False

                # First hour of the day: 4 hours. Time to calculate the asian range and start working!.
                if str(cassie.local_aware.hour) == cassie.frankfurt_open and cassie.local_aware.minute == 0:
                    print('First hour of the day. Time to calculate the asian range and start working!')
                    print('Checking the crypto: ' + coin['symbol'])
                    dataset = cassie.client.get_klines(symbol=coin['symbol'], interval=cassie.WORK_TIMEFRAME,
                                                       limit=cassie.NO_DATA)
                    dataframe = Datapreparation.OHLCV_DataFrame(dataset)
                    upper_limit, lower_limit = AsianRange.update(coin['symbol'], dataframe)

                #fix tabulations (1 less)
                #print(data_returns)
                assets, optimal_sharpe_weights, optimal_variance_weights = Portfolio.Plot(data_returns)
                Portfolio.Optimum_Accounting(assets, optimal_variance_weights, coin['symbol'])


while True:
    if __name__ == "__main__":
        main()
