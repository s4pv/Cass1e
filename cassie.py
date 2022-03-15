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
from preprocessing import Preprocessing
from machinelearning import MachineLearning
from modelforecast import ModelForecast
from finance import Finance
from stats import Stats
from modelfit import ModelFit
import pandas as pd

warnings.filterwarnings("ignore")


class Cassie:

    print('----------------------------------')
    print('|       Welcome to Cass1e        |')
    print('----------------------------------')

    # Configuration and class variables
    parsed_config = Helper.load_config('config.yml')

    LOG_TRADES = parsed_config['script_options'].get('LOG_TRADES')
    LOG_FILE = parsed_config['script_options'].get('LOG_FILE')

    NO_DAYS = parsed_config['model_options']['NO_DAYS']

    PAIR_WITH = parsed_config['trading_options']['PAIR_WITH']
    TICKERS_LIST = parsed_config['trading_options']['TICKERS_LIST']
    TIMEFRAME = parsed_config['trading_options']['TIMEFRAME']

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
                # Forcing closing and deleting orders at the end of the new york season. 18 hours localtime.
                dataset = cassie.client.get_klines(symbol=coin['symbol'], interval=cassie.TIMEFRAME,
                                                   limit=cassie.NO_DAYS)
                dataset_ohlcv = Preprocessing.OHLCV_DataFrame(dataset)
                returns = Portfolio.Return(dataset_ohlcv, coin)
                data_returns[coin['symbol']] = returns[coin['symbol']]

                if coin['symbol'] == 'BTCUSDT':

                    #if cassie.local_aware.day == 'MONDAY': #??
                        print('Today is monday. Heaviest day of the week...')
                        print('We have to fit models again, and choose the better to model BTCUSDT.')
                        print('Extracting pricing data from the server...')
                        #dataset = cassie.client.get_klines(symbol=coin['symbol'], interval=cassie.TIMEFRAME,
                        #                                   limit=cassie.NO_DAYS)
                        #dataframe = DataProcessing.OHLCV_DataFrame(dataset)

                        #print(dataframe)

                        print('Running, fitting and comparing all the models. Saving the best data.')
                        #Stats.Shapiro_Wilk(dataset_ohlcv)
                        #names, results = ModelFit.Calculate(dataframe, coin)
                        #MachineLearning.LSTM(dataset_ohlcv, coin)
                        #ModelForecast.Predict_LSTM(dataset_ohlcv, coin)
                        Finance.GARCH(dataset_ohlcv, coin)
                        print('This may take a while...')
                        print('Model chosen!!. So now, we have to fit the data for all the coins in the ticker list.')
                        print('Loading the model on the coin: ' + coin['symbol'])
                        #dataframe[coin['symbol']] = Portfolio.Data(dataframe, coin)

                #print(dataset)
                print('Making a minimum variance portfolio with the forecasted returns.')
                print('So we have to allocate the following ammounts per coin. Cass1e will be trading beetween those ammounts.')
                #print(dataframe)
                print('Compiling the model')
                print('Making forecasts!')
                print('Plotting... ')

                if str(cassie.local_aware.hour) == cassie.new_york_close and cassie.local_aware.minute == 0:
                    print('We ended the new york trading hours. CHecking for additional portfolio re-balancing.')
                    print('Checking the crypto: ' + coin['symbol'])
                    print('Starting rebalance')

                # Entering the ending hours of the NY time from 13 to 18 hours localtime. Time to collect the profit!.
                if cassie.london_close < int(cassie.local_aware.hour) <= cassie.new_york_close:
                    print('We entered the ending hours of the NY timezone. Time for take profit!')
                    print('Checking the balance on the crypto: ' + crypto)
                    print('Trying to take profit')

                # End of the working season, london time: 13 hours. Time to stop working!
                if int(cassie.local_aware.hour) == cassie.london_close and cassie.local_aware.minute == 0:
                    print('We ended the day. That means its time to stop working!')

                # Entering the first hours of the frankfurt/london timezone from 4 to 13 hours. Time to open some.
                if cassie.frankfurt_open < int(cassie.local_aware.hour) < cassie.london_close:
                    print('We entered the trading hours. That means the frankfurt/london timezone.')
                    print('looking to enter the market.')
                    print('Checking the crypto: ' + coin['symbol'])
                    print('Price under the asian range. Looking for long signaling.')

                # First hour of the day: 4 hours. Time to calculate the asian range and start working!.
                if str(cassie.local_aware.hour) == cassie.frankfurt_open and cassie.local_aware.minute == 0:
                    print('First hour of the day. Time to calculate the asian range and start working!')
                    print('Checking the crypto: ' + coin['symbol'])

                    dataset = cassie.client.get_klines(symbol=coin['symbol'], interval=cassie.TIMEFRAME,
                                                       limit=cassie.NO_DAYS)
                    dataframe = Preprocessing.OHLCV_DataFrame(dataset)
                    upper_limit, lower_limit = AsianRange.update(coin['symbol'], dataframe)
    #print(data_returns)
    #port_returns, port_vols = Portfolio.Simulations(data_returns)
    #Portfolio.Efficient_Frontier(data_returns)
    #Portfolio.Plot(data_returns)
    #print(portfolio_alloc)
    #port_return, port_vols, sharpe = Portfolio.Stats(weights, data_returns)
    #optimal_sharpe_weights = Portfolio.Optimize_Sharpe(data_returns)
    #print(optimal_sharpe_weights)
    #optimal_variance_weights = Portfolio.Optimize_Return(data_returns)
    #print(optimal_variance_weights)
    #minimal_volatilities, target_returns = Portfolio.Efficient_Frontier(data_returns, port_returns)
    #Portfolio.Plot(port_returns, port_vols, optimal_sharpe_weights, optimal_variance_weights,
    #                             minimal_volatilities, target_returns)


if __name__ == "__main__":
    main()
