class AsianRange:
    def update(coin, df):
        try:
            upper_limit = max(df['high'])
            lower_limit = min(df['low'])
            print('Established asian range for the crypto: ' + coin)
            print('With an upper limit of:' + upper_limit)
            print('And a lower limit of:' + lower_limit)
        except Exception as e:
            print("An exception occurred - {}".format(e))
            return False
        return upper_limit, lower_limit
