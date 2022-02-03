from binance.client import Client
import keys
import numpy as np
import pandas as pd
import warnings
from dataprocessing import DataProcessing
import matplotlib.pyplot as plt
import time
import scipy.optimize as Optimize
import scipy.interpolate as sci


warnings.filterwarnings("ignore")


class Portfolio:
    def Accounting(crypto):
        try:
            client = Client(keys.Pkey, keys.Skey)
            balance = client.get_asset_balance(crypto)
            if float(balance['free']) > 0:
                print('We are currently bull on: ' + crypto + ' . With a balance of: ' + balance['free'])
            else:
                print('We are bearish on:' + crypto + '  .With a balance of: ' + balance['free'])
            usdt_balance = client.get_asset_balance('USDT')
            if float(balance['free']) > 0:
                print('We are sitting on: ' + balance['free'] + ' USDT.')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return balance['free']

    def Return(dataset, coin):
        try:
            dataframe = DataProcessing.Reshape_Data(dataset, 'NO_MODEL')
            dataframe = pd.DataFrame(dataframe)
            # dataframe[coin['symbol']] = dataframe / dataframe.shift(1) - 1
            dataframe[coin['symbol']] = np.log(dataframe/dataframe.shift(1))
            dataframe.pop(0)
            dataframe = dataframe.drop(dataframe.index[0])
            if len(dataframe) == 999:
                df = dataframe
            elif len(dataframe) < 999:
                shift = 999 - len(dataframe) + 1
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
                port_returns.append(np.sum(data_returns.mean() * weights) * 252)
                port_vols.append(np.sqrt(np.dot(weights.T, np.dot(data_returns.cov() * 252, weights))))

            # Convert lists to arrays
            port_returns = np.array(port_returns)
            port_vols = np.array(port_vols)

            # Plot the distribution of portfolio returns and volatilities
            plt.figure(figsize=(18, 10))
            plt.scatter(port_vols, port_returns, c=(port_returns / port_vols), marker='o')
            plt.xlabel('Portfolio Volatility')
            plt.ylabel('Portfolio Return')
            plt.colorbar(label='Sharpe ratio (not adjusted for short rate)')
            plt.show()
            print('Elapsed Time: %.2f seconds' % (time.time() - start))
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return port_returns, port_vols

    def Stats(data_returns, weights):
        try:
            # Convert to array in case list was passed instead.
            weights = np.array(weights)
            # Anualize: 252
            port_return = np.sum(data_returns.mean() * weights) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(data_returns.cov() * 252, weights)))
            sharpe = port_return / port_vol
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}

    def minimize_sharpe(data_returns, weights):
        return -Portfolio.Stats(data_returns, weights)['sharpe']

    def minimize_volatility(data_returns, weights):
        return Portfolio.Stats(data_returns, weights)['volatility']

    def minimize_return(data_returns, weights):
        return -Portfolio.Stats(data_returns, weights)['return']

    def Optimize_Sharpe(data_returns):
        try:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            num_assets = len(data_returns.columns)
            assets = data_returns.columns
            weights = np.random.dirichlet(np.ones(num_assets), size=1)
            weights = weights[0]
            bounds = tuple((0, 1) for x in range(num_assets))
            initializer = num_assets * [1. / num_assets, ]
            print(initializer)
            print(bounds)
            optimal_sharpe = Optimize.minimize(Portfolio.minimize_sharpe(data_returns, weights),
                                               initializer,
                                               method='SLSQP',
                                               bounds=bounds,
                                               constraints=constraints)
            print(optimal_sharpe)
            optimal_sharpe_weights = optimal_sharpe['x'].round(4)
            list(zip(assets, list(optimal_sharpe_weights)))
            optimal_stats = Portfolio.Stats(data_returns, optimal_sharpe_weights)
            print('Optimal Portfolio Return: ', round(optimal_stats['return'] * 100, 4))
            print('Optimal Portfolio Volatility: ', round(optimal_stats['volatility'] * 100, 4))
            print('Optimal Portfolio Sharpe Ratio: ', round(optimal_stats['sharpe'], 4))

        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return list(zip(assets, list(optimal_sharpe_weights)))

    def Optimize_Return(data_returns):
        try:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            num_assets = len(data_returns.columns)
            assets = data_returns.columns
            weights = np.random.dirichlet(np.ones(num_assets), size=1)
            weights = weights[0]
            bounds = tuple((0, 1) for x in range(num_assets))
            initializer = num_assets * [1. / num_assets, ]
            optimal_variance = Optimize.minimize(Portfolio.minimize_volatility(data_returns, weights),
                                                 initializer,
                                                 method='SLSQP',
                                                 bounds=bounds,
                                                 constraints=constraints)
            optimal_variance_weights=optimal_variance['x'].round(4)
            list(zip(assets, list(optimal_variance_weights)))
            optimal_stats = Portfolio.Stats(data_returns, optimal_variance_weights)

            print(optimal_stats)
            print('Optimal Portfolio Return: ', round(optimal_stats['return'] * 100, 4))
            print('Optimal Portfolio Volatility: ', round(optimal_stats['volatility'] * 100, 4))
            print('Optimal Portfolio Sharpe Ratio: ', round(optimal_stats['sharpe'], 4))

        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return list(zip(assets, list(optimal_variance_weights)))

    def Efficient_Frontier(data_returns, port_returns):
        try:
            # Make an array of 50 returns between the minimum return and maximum return
            # discovered earlier.
            target_returns = np.linspace(port_returns.min(), port_returns.max(), 50)
            num_assets = len(data_returns.columns)
            weights = np.random.dirichlet(np.ones(num_assets), size=1)
            weights = weights[0]
            # Initialize optimization parameters
            minimal_volatilities = []
            bounds = tuple((0, 1) for x in weights)
            initializer = num_assets * [1. / num_assets, ]

            for target_return in target_returns:
                constraints = ({'type': 'eq', 'fun': lambda x: Portfolio.Stats(x, weights)['return'] - target_return},
                               {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

                optimal = Optimize.minimize(Portfolio.minimize_volatility(data_returns, weights),
                                            initializer,
                                            method='SLSQP',
                                            bounds=bounds,
                                            constraints=constraints)

                minimal_volatilities.append(optimal['fun'])

            minimal_volatilities = np.array(minimal_volatilities)
            target_returns = np.array(target_returns) #ver si es con S o no final

        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return minimal_volatilities, target_returns

    def Plot(data_returns, port_returns, port_vols, optimal_sharpe_weights, optimal_variance_weights,
             minimal_volatilities, target_returns):
        try:
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

            plt.plot(Portfolio.Stats(data_returns, optimal_sharpe_weights)['volatility'],
                     Portfolio.Stats(data_returns, optimal_sharpe_weights)['return'],
                     'r*',
                     markersize=25.0)

            plt.plot(Portfolio.Stats(data_returns, optimal_variance_weights)['volatility'],
                     Portfolio.Stats(data_returns, optimal_variance_weights)['return'],
                     'y*',
                     markersize=25.0)

            plt.xlabel('Portfolio Volatility')
            plt.ylabel('Portfolio Return')
            plt.colorbar(label='Sharpe ratio (not adjusted for short rate)')

            plt.show()
        except Exception as e:
            print("An exception ocurred - {}".format(e))
            return False
        return True

    def Capital_Market_Line(minimal_volatilities, target_returns, port_vols, port_returns):
        try:
            min_index = np.argmin(minimal_volatilities)
            ex_returns = target_returns[min_index:]
            ex_volatilities = minimal_volatilities[min_index:]

            var = sci.splrep(ex_returns, ex_volatilities)
            rfr = 0.01
            m = port_vols.max() / 2
            li = port_returns.max() / 2

            optimal = Optimize.fsolve(Portfolio.eqs, [rfr, m, li])
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
