class AsianRange:
    def update(coin, df):
        try:
            upper_limit = df['high'][len(df)-1]
            lower_limit = df['low'][(len(df))-1]
            for x in range(len(df) - 10 - 1, len(df) - 1): #range choosen based on a work timeframe of 1hour
                if df['high'][x] > upper_limit:
                    upper_limit = df['high'][x]
                if df['low'][x] < lower_limit:
                    lower_limit = df['low'][x]
            print('Established asian range for the crypto: ' + coin)
            print('With an upper limit of:' + upper_limit)
            print('And a lower limit of:' + lower_limit)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return upper_limit, lower_limit
