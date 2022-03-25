class Tradesignals:
    def Looking_Long(crypto, df):
        try:
            if df['close'][len(df)-3] < df['open'][len(df)-3] and df['open'][len(df)-2] < df['close'][len(df)-2]:
                Buy_Sign = True
                print('Buy signal came true on crypto pair: ' + crypto)
            else:
                Buy_Sign = False
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return Buy_Sign

    def Looking_Short(crypto, df):
        try:
            if df['close'][len(df)-3] > df['open'][len(df)-3] and df['open'][len(df)-2] > df['close'][len(df)-2]:
                Sell_Sign = True
                print('Sell signal came true on crypto pair: ' + crypto)
            else:
                Sell_Sign = False
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return Sell_Sign