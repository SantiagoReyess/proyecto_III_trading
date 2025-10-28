from dataclasses import dataclass


def backtesting(dataframe, stop_loss, take_profit, n_shares):

    """
    Simulates a trading strategy on historical data (backtesting).

    This function iterates through a pandas DataFrame containing price data and
    trading signals. It executes long and short trades, manages open positions
    with stop-loss and take-profit levels, and tracks the portfolio's value
    over time. It accounts for commissions on trades and borrowing costs for
    short positions.

    Args:
        dataframe (pd.DataFrame): A DataFrame with historical market data.
            It must contain the following columns:
            - 'Price': The price at each time step.
            - 'buy_signal': A boolean column, True to open a long position.
            - 'sell_signal': A boolean column, True to open a short position.
        stop_loss (float): The percentage (as a decimal, e.g., 0.02 for 2%)
            below the entry price to close a long position, or above for a
            short position.
        take_profit (float): The percentage (as a decimal, e.g., 0.05 for 5%)
            above the entry price to close a long position, or below for a
            short position.
        n_shares (int): The number of shares to trade in each operation.

    Returns:
        list: A list of floats representing the portfolio's total value at
              each time step of the simulation.
    """


    cash = 10000
    COM =  0.125/100
    BORROW_RATE = 0.25 / 100 / 252

    @dataclass
    class Operation:
        price: float
        sl: float
        tp: float
        n_shares: int

    active_long_positions = []
    active_short_positions = []

    portfolio_historic = [cash]

    data = dataframe.copy()

    for i, row in data.iterrows():

        # Close Long Positions
        for pos in active_long_positions.copy():

            if (pos.sl > row.Price) or (pos.tp < row.Price):
                cash += row.Price * n_shares * (1 - COM)
                active_long_positions.remove(pos)

        # Close Short Positions
        for pos in active_short_positions.copy():

            if (pos.sl < row.Price) or (pos.tp > row.Price):
                pnl = (pos.price - row.Price) * n_shares
                commision = row.Price * n_shares * COM
                cash += pnl - commision
                active_short_positions.remove(pos)

        for pos in active_short_positions.copy():
            cash -= row.Price * pos.n_shares * BORROW_RATE

        # Open Long Positions
        if True == row.buy_signal:
            cost = row.Price * n_shares * (1+ COM)

            if cash > cost:
                cash -= cost

                active_long_positions.append(Operation
                                             (price = row.Price,
                                              n_shares = n_shares,
                                              sl = row.Price * (1 - stop_loss),
                                              tp = row.Price * (1 + take_profit)))

        # Open Short Positions
        if True == row.sell_signal:
            cost = row.Price * n_shares * COM

            if cash > cost:
                cash -= cost
                active_short_positions.append(Operation
                                              (price = row.Price,
                                               n_shares = n_shares,
                                               sl = row.Price * (1 + stop_loss),
                                               tp = row.Price * (1 -  take_profit)))

        # Value Portfolio for each row
        portfolio_val = 0
        portfolio_val += cash

        ## Value Long positions
        for pos in active_long_positions.copy():
            portfolio_val += row.Price * pos.n_shares

        ## Value Short Positions
        for pos in active_short_positions.copy():
            portfolio_val += (pos.price * n_shares) - (row.Price * n_shares)

        # Add portfolio value to historic
        portfolio_historic.append(portfolio_val)

    last_close = data["Price"].iloc[-1]

    ## Close ALL Long Positions
    for pos in active_long_positions.copy():
        pnl = (pos.price - last_close) * n_shares
        commision = last_close * n_shares * COM
        cash += pnl - commision
        active_long_positions.remove(pos)

    ## Close ALL Short Positions
    for pos in active_short_positions.copy():
        cash += (pos.price * n_shares) - (last_close * n_shares * (1 + COM))
        active_short_positions.remove(pos)

    portfolio_val = cash
    portfolio_historic.append(portfolio_val)

    return portfolio_historic



