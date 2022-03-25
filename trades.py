from binance.client import Client
import keys

class Trades:

    def MarketBuy(coin, qtty, open_price):
        try:
            client = Client(keys.Pkey, keys.Skey)
            orders = client.get_all_orders(symbol=coin['symbol'])
            buy_limit_count = 0
            for x in range(len(orders)): #seems that also can work: for x in orders.items():
                if orders[x]['type'] == 'LIMIT':
                    if orders[x]['side'] == 'BUY':
                        if orders[x]['status'] == 'NEW':
                            buy_limit_count += 1
            if buy_limit_count == 0:
                client.order_limit_sell(coin['symbol'], quantity=qtty, price=open_price)
                print('Send buy limit on the crypto: ' + coin['symbol'] + ' .')
                print('Size: ' + qtty + ' .')
                print('At the price: ' + open_price + ' .')
                buy_limit_count += 1
            print('We currently have: ' + str(buy_limit_count) + ' buy limit(s).')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True


    def MarketSell(coin, qtty, open_price):
        try:
            client = Client(keys.Pkey, keys.Skey)
            orders = client.get_all_orders(symbol=coin['symbol'])
            sell_limit_count = 0
            for x in range(len(orders)):
                if orders[x]['type'] == 'LIMIT':
                    if orders[x]['side'] == 'SELL':
                        if orders[x]['status'] == 'NEW':
                            sell_limit_count += 1
            if sell_limit_count == 0:
                client.order_limit_sell(coin['symbol'], quantity=qtty, price=open_price)
                print('Send buy limit on the crypto: ' + coin['symbol'] + ' .')
                print('Size: ' + qtty + ' .')
                print('At the price: ' + open_price + ' .')
                sell_limit_count += 1
            print('We currently have: ' + str(sell_limit_count) + ' sell limit(s).')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True


    def LimitBuy(coin, qtty, open_price):
        try:
            client = Client(keys.Pkey, keys.Skey)
            orders = client.get_all_orders(symbol=coin['symbol'])
            buy_limit_count = 0
            for x in range(len(orders)): #seems that also can work: for x in orders.items():
                if orders[x]['type'] == 'LIMIT':
                    if orders[x]['side'] == 'BUY':
                        if orders[x]['status'] == 'NEW':
                            buy_limit_count += 1
            if buy_limit_count == 0:
                client.order_limit_sell(coin['symbol'], quantity=qtty, price=open_price)
                print('Send buy limit on the crypto: ' + coin['symbol'] + ' .')
                print('Size: ' + qtty + ' .')
                print('At the price: ' + open_price + ' .')
                buy_limit_count += 1
            print('We currently have: ' + str(buy_limit_count) + ' buy limit(s).')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True


    def LimitSell(coin, qtty, open_price):
        try:
            client = Client(keys.Pkey, keys.Skey)
            orders = client.get_all_orders(symbol=coin['symbol'])
            sell_limit_count = 0
            for x in range(len(orders)):
                if orders[x]['type'] == 'LIMIT':
                    if orders[x]['side'] == 'SELL':
                        if orders[x]['status'] == 'NEW':
                            sell_limit_count += 1
            if sell_limit_count == 0:
                client.order_limit_sell(coin['symbol'], quantity=qtty, price=open_price)
                print('Send buy limit on the crypto: ' + coin['symbol'] + ' .')
                print('Size: ' + qtty + ' .')
                print('At the price: ' + open_price + ' .')
                sell_limit_count += 1
            print('We currently have: ' + str(sell_limit_count) + ' sell limit(s).')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True


    def Delete_Order(coin):
        try:
            client = Client(keys.Pkey, keys.Skey)
            orders = client.get_all_orders(symbol=coin['symbol'])
            for x in range(len(orders)):
                if orders[x]['type'] == 'LIMIT':
                    if orders[x]['status'] == 'NEW':
                            client.cancel_order(coin['symbol'], orders[x]['orderId'])
            print('Deleted limit order: ' + str(orders[x]['orderId']) + ' .')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True