from binance.client import Client
import keys
import numpy as np
import pandas as pd
import warnings
from datapreparation import Datapreparation
import matplotlib.pyplot as plt
import time
import scipy.optimize as optimize
import scipy.interpolate as sci
from helper import Helper
import os

warnings.filterwarnings("ignore")

parsed_config = Helper.load_config('config.yml')
N_STEPS_OUT = parsed_config['ml_options']['N_STEPS_OUT']
NO_DATA = parsed_config['model_options']['NO_DATA']

class Portfolio:
    def Accounting(crypto):
        try:
            client = Client(keys.Pkey, keys.Skey)
            balance = client.get_asset_balance(crypto)
            if float(balance['free']) > 0:
                print('We are currently bull on: ' + crypto + ' . With a balance of: ' + balance['free'])
            usdt_balance = client.get_asset_balance('USDT')
            if float(usdt_balance) > 0:
                print('We are sitting on: ' + usdt_balance + ' USDT.')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return balance['free'], usdt_balance

    def Optimum_Accounting(assets, assets_weights, crypto):
        try:
            pass
            #for x in len(assets):
                #if x == crypto:
            #       port_size = np.append(assets[x], assets_weights[x])
            #print(port_opt)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True

    def Return(dataset, coin):
        try:
            dataframe = pd.DataFrame(dataset)
            dataframe = Datapreparation.Reshape_Float(dataframe)
            dataframe = pd.DataFrame(dataframe)
            # ratio returns
            # dataframe[coin['symbol']] = dataframe / dataframe.shift(1) - 1
            # log returns
            dataframe[coin['symbol']] = np.log(dataframe/dataframe.shift(1))
            dataframe.pop(0)
            dataframe = dataframe.drop(dataframe.index[0])
            if len(dataframe) == NO_DATA + N_STEPS_OUT - 1:
                df = dataframe
            elif len(dataframe) < NO_DATA + N_STEPS_OUT - 1:
                shift = NO_DATA + N_STEPS_OUT - 1 - len(dataframe) + 1
                df = pd.DataFrame(columns=[coin['symbol']], index=range(shift))
                df = pd.concat([df, dataframe], ignore_index=True)
                df[coin['symbol']] = df[coin['symbol']].fillna(0)
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return df

    def Simulations(data_returns):
        try:
            port_returns = []
            port_vols = []
            iterations = 3000
            num_assets = len(data_returns.columns)
            start = time.time()
            for i in range(iterations):
                weights = np.random.dirichlet(np.ones(num_assets), size=1)
                weights = weights[0]
                # Anualize: 252
                #port_returns.append(np.sum(data_returns.mean() * weights) * 252)
                #port_vols.append(np.sqrt(np.dot(weights.T, np.dot(data_returns.cov() * 252, weights))))
                # no Anualize
                port_returns.append(np.sum(data_returns.mean() * weights))
                port_vols.append(np.sqrt(np.dot(weights.T, np.dot(data_returns.cov(), weights))))
            # Convert lists to arrays
            port_returns = np.array(port_returns)
            port_vols = np.array(port_vols)
            # Plot the distribution of portfolio returns and volatilities
            plt.figure(figsize=(18, 10))
            plt.scatter(port_vols, port_returns, c=(port_returns / port_vols), marker='o')
            plt.xlabel('Portfolio Volatility')
            plt.ylabel('Portfolio Return')
            plt.colorbar(label='Sharpe ratio (not adjusted for short rate)')
            #plt.show()
            filename = os.path.join('portfolio_plots/simulations.png')
            plt.savefig(filename)
            plt.close()
            print('Elapsed Time: %.2f seconds' % (time.time() - start))
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return port_returns, port_vols

    def port_stats(weights, data_returns):
        try:
            # Convert to dataframe in case list was passed instead.
            df = pd.DataFrame(data_returns)
            # Convert to array in case list was passed instead.
            weights = np.array(weights)
            # Anualize: 252
            #port_return = np.sum(df.mean() * weights)* 252
            #port_vol = np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))
            # no Anualize
            port_return = np.sum(df.mean() * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(df.cov(), weights)))
            sharpe = port_return / port_vol
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}

    def minimize_sharpe(weights, data_returns):
        try:
            ms = -Portfolio.port_stats(weights, data_returns)['sharpe']
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return ms

    def minimize_volatility(weights, data_returns):
        try:
            mv = Portfolio.port_stats(weights, data_returns)['volatility']
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return mv

    def minimize_return(weights, data_returns):
        try:
            mr = -Portfolio.port_stats(weights, data_returns)['return']
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return mr

    def Optimize_Sharpe(data_returns):
        try:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            num_assets = len(data_returns.columns)
            assets = data_returns.columns
            bounds = tuple((0, 1) for x in range(num_assets))
            initializer = num_assets * [1. / num_assets, ]
            optimal_sharpe = optimize.minimize(fun=Portfolio.minimize_sharpe,
                                               x0=initializer,
                                               args=data_returns,
                                               method='SLSQP',
                                               bounds=bounds,
                                               constraints=constraints)
            optimal_sharpe_weights = optimal_sharpe['x'].round(4)
            optimal_stats = Portfolio.port_stats(optimal_sharpe_weights, data_returns)
            print('Optimal Sharpe Results:')
            print('Optimal Portfolio Return: ', round(optimal_stats['return'] * 100, 4))
            print('Optimal Portfolio Volatility: ', round(optimal_stats['volatility'] * 100, 4))
            print('Optimal Portfolio Sharpe Ratio: ', round(optimal_stats['sharpe'], 4))
            print(list(zip(assets, list(optimal_sharpe_weights))))
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return list(zip(assets, list(optimal_sharpe_weights)))

    def Optimize_Return(data_returns):
        try:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            num_assets = len(data_returns.columns)
            assets = data_returns.columns
            bounds = tuple((0, 1) for x in range(num_assets))
            initializer = num_assets * [1. / num_assets, ]
            optimal_variance = optimize.minimize(Portfolio.minimize_volatility,
                                                 x0=initializer,
                                                 args=data_returns,
                                                 method='SLSQP',
                                                 bounds=bounds,
                                                 constraints=constraints)
            optimal_variance_weights=optimal_variance['x'].round(4)
            optimal_stats = Portfolio.port_stats(optimal_variance_weights, data_returns)
            print('Minimum Variance results:')
            print('Optimal Portfolio Return: ', round(optimal_stats['return'] * 100, 4))
            print('Optimal Portfolio Volatility: ', round(optimal_stats['volatility'] * 100, 4))
            print('Optimal Portfolio Sharpe Ratio: ', round(optimal_stats['sharpe'], 4))
            print(list(zip(assets, list(optimal_variance_weights))))
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return list(zip(assets, list(optimal_variance_weights)))

    def Efficient_Frontier(data_returns):
        try:
            num_assets = len(data_returns.columns)
            weights = np.random.dirichlet(np.ones(num_assets), size=1)
            weights = weights[0]
            # Initialize optimization parameters
            minimal_volatilities = []
            bounds = tuple((0, 1) for x in weights)
            initializer = num_assets * [1. / num_assets, ]
            # Make an array of 50 returns between the minimum return and maximum return
            # discovered earlier.
            port_returns, port_vols = Portfolio.Simulations(data_returns)
            target_returns = np.linspace(port_returns.min(), port_returns.max(), 50)
            for target_return in target_returns:
                constraints = ({'type': 'eq', 'fun': lambda x: Portfolio.port_stats(x, data_returns)['return'] - target_return},
                               {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                optimal = optimize.minimize(fun=Portfolio.minimize_volatility,
                                            x0=initializer,
                                            args=data_returns,
                                            method='SLSQP',
                                            bounds=bounds,
                                            constraints=constraints)
                minimal_volatilities.append(optimal['fun'])
            minimal_volatilities = np.array(minimal_volatilities)
            target_returns = np.array(target_returns)
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return minimal_volatilities, target_returns

    def Plot(data_returns):
        try:
            port_returns, port_vols = Portfolio.Simulations(data_returns)
            minimal_volatilities, target_returns = Portfolio.Efficient_Frontier(data_returns)
            #print(minimal_volatilities)
            #print(target_returns)
            sharpe_results = Portfolio.Optimize_Sharpe(data_returns)
            assets1, optimal_sharpe_weights = list(zip(*sharpe_results))
            var_results = Portfolio. Optimize_Return(data_returns)
            assets2, optimal_variance_weights = list(zip(*var_results))
            # initialize figure size
            plt.figure(figsize=(18, 10))

            plt.scatter(port_vols,
                        port_returns,
                        c=(port_returns / port_vols),
                        marker='o')

            plt.scatter(minimal_volatilities,
                        target_returns,
                        c=(target_returns / minimal_volatilities),
                        marker='x')

            plt.plot(Portfolio.port_stats(optimal_sharpe_weights, data_returns)['volatility'],
                     Portfolio.port_stats(optimal_sharpe_weights, data_returns)['return'],
                     'r*',
                     markersize=25.0)

            plt.plot(Portfolio.port_stats(optimal_variance_weights, data_returns)['volatility'],
                     Portfolio.port_stats(optimal_variance_weights, data_returns)['return'],
                     'y*',
                     markersize=25.0)

            plt.xlabel('Portfolio Volatility')
            plt.ylabel('Portfolio Return')
            plt.colorbar(label='Sharpe ratio (not adjusted for short rate)')
            #plt.show()
            filename = os.path.join('portfolio_plots/Efficient_Frontier.png')
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return assets1, optimal_sharpe_weights, optimal_variance_weights

#There is missing sintax below here
    def Capital_Market_Line(data_returns):
        try:
            minimal_volatilities, target_returns = Portfolio.Efficient_Frontier(data_returns)
            port_returns, port_vols = Portfolio.Simulations(data_returns)
            min_index = np.argmin(minimal_volatilities)
            ex_returns = target_returns[min_index:]
            ex_volatilities = minimal_volatilities[min_index:]

            var = sci.splrep(ex_returns, ex_volatilities)
            rfr = 0.01
            m = port_vols.max() / 2
            li = port_returns.max() / 2

            optimal = optimize.fsolve(Portfolio.eqs, [rfr, m, li])
            print(optimal)
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return True

    def func(x, var):
        try:
            # Spline approximation of the efficient frontier
            spline_approx = sci.splev(x, var, der=0)
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return spline_approx

    def d_func(x, var):
        try:
            # first derivative of the approximate efficient frontier function
            deriv = sci.splev(x, var, der=1)
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return deriv

    def eqs(p, rfr=0.01):
        try:
            # rfr = risk free rate

            eq1 = rfr - p[0]
            eq2 = rfr + p[1] * p[2] - Portfolio.func(p[2])
            eq3 = p[1] - Portfolio.d_func(p[2])
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return eq1, eq2, eq3
