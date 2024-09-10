import alpaca_trade_api as tradeapi
import logging as lg
import sys
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sentiment_analysis import fetch_news_articles, analyze_sentiment

# Load configuration from config.json
def load_config():
    config_path = r"C:\Users\Asus\Desktop\trading-bot\config.json"  # Ensure the path is correct
    with open(config_path) as config_file:
        return json.load(config_file)

# Load the configuration
config = load_config()

# Initialize Alpaca API (using paper trading environment)
api = tradeapi.REST(config['api_key'], config['api_secret'], base_url='https://paper-api.alpaca.markets')

class Trader:
    def __init__(self, ticker):
        self.ticker = ticker
        lg.info(f'Trader initialized with ticker {ticker}')

    def get_historical_data(self, months=18):
        """
        Fetch OHLC data for the past X months (default: 18 months for more data).
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')

            # Fetch historical OHLC data
            bars = api.get_bars(
                self.ticker,
                tradeapi.rest.TimeFrame.Day,
                start=start_date,
                end=end_date,
                limit=100,
                adjustment='raw',
                feed='iex'
            )

            df = pd.DataFrame({
                'Open': [bar.o for bar in bars],
                'High': [bar.h for bar in bars],
                'Low': [bar.l for bar in bars],
                'Close': [bar.c for bar in bars],
                'Volume': [bar.v for bar in bars]
            })

            if df.empty:
                lg.error(f"No OHLC data available for {self.ticker}.")
                sys.exit()

            return df
        except Exception as e:
            lg.error(f'Error fetching OHLC data: {e}')
            sys.exit()

    def plot_stock_data(self, df):
        """
        Plot the stock prices (OHLC) for the past year.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Open'], label='Open', color='blue', alpha=0.7)
        plt.plot(df.index, df['High'], label='High', color='green', alpha=0.7)
        plt.plot(df.index, df['Low'], label='Low', color='red', alpha=0.7)
        plt.plot(df.index, df['Close'], label='Close', color='black', alpha=0.7)

        plt.title(f'{self.ticker} Stock Prices - Last 12 Months')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

    def calculate_moving_average(self, df, period=50):
        if len(df) < period:
            lg.info(f"Not enough data to calculate {period}-day moving average.")
            return df['Close'].rolling(window=len(df)).mean().iloc[-1]  # Use available data
        return df['Close'].rolling(window=period).mean().iloc[-1]

    def check_moving_average_crossover(self, df):
        short_term_ma = df['Close'].rolling(window=50).mean()
        long_term_ma = df['Close'].rolling(window=200).mean()

        if short_term_ma.iloc[-1] > long_term_ma.iloc[-1] and short_term_ma.iloc[-2] <= long_term_ma.iloc[-2]:
            return "Golden Cross - Buy Signal"
        elif short_term_ma.iloc[-1] < long_term_ma.iloc[-1] and short_term_ma.iloc[-2] >= long_term_ma.iloc[-2]:
            return "Death Cross - Sell Signal"
        return "No Crossover Detected"

    def calculate_macd(self, df):
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()

        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return "MACD Bullish Crossover - Buy Signal"
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return "MACD Bearish Crossover - Sell Signal"
        return "No MACD Crossover Detected"

    def calculate_bollinger_bands(self, df):
        mid_band = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        upper_band = mid_band + (std * 2)
        lower_band = mid_band - (std * 2)

        return upper_band, lower_band

    def calculate_bollinger_band_width(self, df):
        """
        Calculate the Bollinger Band width.
        """
        upper_band, lower_band = self.calculate_bollinger_bands(df)
        band_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / lower_band.iloc[-1]
        return band_width

    def calculate_adx(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']

        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        return adx.iloc[-1]

    def check_volume_spike(self, df):
        if len(df) < 30:
            lg.info("Not enough data to calculate 30-day volume average.")
            return False

        avg_volume = df['Volume'].rolling(window=30).mean().iloc[-1]
        latest_volume = df['Volume'].iloc[-1]
        return latest_volume >= 1.15 * avg_volume

    def fetch_and_analyze_news(self):
        """
        Fetch the last 5 news articles about the company, display them, and analyze their sentiment.
        """
        lg.info(f'Fetching news articles for {self.ticker}...')
        news_articles = fetch_news_articles(self.ticker)
        
        if news_articles:
            sentiment_score = 0
            print("\nLatest 5 News Articles and Their Sentiment Scores:")
            for article in news_articles:
                article_sentiment = analyze_sentiment(article)
                sentiment_score += article_sentiment
                print(f"Article: {article}\nSentiment Score: {article_sentiment}\n")

            lg.info(f"Overall sentiment score for {self.ticker}: {sentiment_score}")
            return sentiment_score
        else:
            lg.info(f"No news articles found for {self.ticker}.")
            return 0

    def should_buy(self, df):
        """
        Decision-making logic with updated conditions and dynamic thresholding.
        """
        last_close = df['Close'].iloc[-1]
        sentiment_score = self.fetch_and_analyze_news()

        # Moving averages crossover
        ma_crossover = self.check_moving_average_crossover(df)
        lg.info(ma_crossover)

        # MACD
        macd_signal = self.calculate_macd(df)
        lg.info(macd_signal)

        # Bollinger Bands
        upper_band, lower_band = self.calculate_bollinger_bands(df)
        if last_close > upper_band.iloc[-1]:
            lg.info(f"Bollinger Bands Breakout (Above Upper Band) - Potential Buy Signal")
        elif last_close < lower_band.iloc[-1]:
            lg.info(f"Bollinger Bands Breakout (Below Lower Band) - Potential Sell Signal")

        # ADX (Trend Strength)
        adx_value = self.calculate_adx(df)
        lg.info(f"ADX Value: {adx_value} - Trend Strength")

        # Volume Spike
        volume_spike = self.check_volume_spike(df)
        lg.info(f"Volume Spike: {volume_spike}")

        # Calculate Bollinger Band Width
        band_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / upper_band.iloc[-1]
        lg.info(f"Bollinger Band Width: {band_width}")

        # Dynamic threshold based on volatility (Bollinger Band Width)
        dynamic_threshold = 0.5 if band_width > 0.05 else 0.6
        lg.info(f"Dynamic threshold based on volatility: {dynamic_threshold * 100}%")

        # Weightage system
        sentiment_weight = 0.2 if sentiment_score >= 3 else 0.0
        ma_weight = 0.15 if "Golden Cross" in ma_crossover else 0.0
        macd_weight = 0.15 if "Bullish Crossover" in macd_signal else 0.0
        adx_weight = 0.1 if adx_value >= 25 else 0.0
        volume_weight = 0.1 if volume_spike else 0.0

        # Calculate the overall score
        overall_score = sentiment_weight + ma_weight + macd_weight + adx_weight + volume_weight

        lg.info(f"Overall Score: {overall_score * 100}%")

        # Provide detailed output for each weightage
        lg.info(f"Sentiment Weight: {sentiment_weight * 100}%")
        lg.info(f"Moving Average Crossover Weight: {ma_weight * 100}%")
        lg.info(f"MACD Weight: {macd_weight * 100}%")
        lg.info(f"ADX Weight: {adx_weight * 100}%")
        lg.info(f"Volume Spike Weight: {volume_weight * 100}%")

        # Use dynamic threshold to decide whether to buy
        if overall_score >= dynamic_threshold:
            lg.info(f"BUY signal for {self.ticker}. All conditions met.")
            print(f"BUY signal for {self.ticker}.")
        else:
            lg.info(f"NO BUY signal for {self.ticker}. Conditions not met.")
            print(f"NO BUY signal for {self.ticker}.")

    def get_account_info(self):
        """
        Fetch account information using Alpaca API.
        """
        try:
            account = api.get_account()
            lg.info(f'Account balance: {account.cash}')
            return account
        except Exception as e:
            lg.error(f'Error fetching account info: {e}')
            sys.exit()
