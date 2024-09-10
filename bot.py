from traderLib import *
from logger import *

# Initialize the logger
initialise_logger()

# Check trading account
def check_account_ok():
    try:
        trader = Trader('AAPL')  # Example ticker symbol
        account = trader.get_account_info()
        if float(account.cash) < 100:  # Example: stop if account balance is less than $100
            lg.error("Insufficient account balance")
            sys.exit()
    except Exception as e:
        lg.error("Could not get account info")
        lg.error(str(e))
        sys.exit()

# Define asset
def get_ticker():
    # Ask for ticker from user
    ticker = input('Write the ticker you want to operate with: ')
    return ticker

# Execute trading bot
# Execute trading bot
def main():
    # Initialize logger
    initialise_logger()

    # Check account balance
    check_account_ok()

    # Get stock ticker
    ticker = get_ticker()

    # Initialize trader
    trader = Trader(ticker)

    # Get 100-day historical OHLC data
    df = trader.get_historical_data()

    # Display OHLC data
    print(f"OHLC data for {ticker} over the last 100 days:\n{df}\n")

    # **Plot the stock data for the last year**
    trader.plot_stock_data(df)  # Ensure this is called to display the graph.

    # Decide if you should buy
    if trader.should_buy(df):
        print(f"BUY signal for {ticker}.")
    else:
        print(f"NO BUY signal for {ticker}.")


if __name__ == "__main__":
    main()
