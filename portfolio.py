from binance.client import Client
import keys


class Portfolio:
    def accounting(crypto):
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

    def optimization(self):
        try:
            pass
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return True
