from binance.client import Client
import keys


class Trades:
    def Long(coin):
        try:
            client = Client(keys.Pkey, keys.Skey)
            orders = client.get_all_orders(symbol=coin['symbol'])
            buy_limit_count = 0
            for x in range(len(orders)): #seems that also can work: for x in orders.items():
                if orders[x]['type'] == 'LIMIT':
                    if orders[x]['side'] == 'BUY':
                        if orders[x]['status'] == 'NEW':
                            buy_limit_count += 1
            print('We currently have: ' + str(buy_limit_count) + ' buy limit(s).')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return buy_limit_count


    def Short(coin):
        try:
            client = Client(keys.Pkey, keys.Skey)
            orders = client.get_all_orders(symbol=coin['symbol'])
            sell_limit_count = 0
            for x in range(len(orders)):
                if orders[x]['type'] == 'LIMIT':
                    if orders[x]['side'] == 'SELL':
                        if orders[x]['status'] == 'NEW':
                            sell_limit_count += 1
            print('We currently have: ' + str(sell_limit_count) + ' sell limit(s).')
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return sell_limit_count