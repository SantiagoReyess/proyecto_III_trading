from dataclasses import dataclass


def backtesting(dataframe, stop_loss, take_profit, n_shares):

    cash = 1000000
    COM =  0.125/100
    BORROW_RATE = 0.25 / 100

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

            if (pos.sl > row.Close) or (pos.tp < row.Close):
                cash += row.Close * n_shares * (1 - COM)
                active_long_positions.remove(pos)

        # Close Short Positions
        for pos in active_short_positions.copy():

            if (pos.sl < row.Close) or (pos.tp > row.Close):
                pnl = (pos.price - row.Close) * n_shares
                commision = row.Close * n_shares * COM
                cash += pnl - commision
                active_short_positions.remove(pos)

        # Open Long Positions
        if True == row.buy_signal:
            cost = row.Close * n_shares * (1+ COM)

            if cash > cost:
                cash -= cost

                active_long_positions.append(Operation
                                             (price = row.Close,
                                              n_shares = n_shares,
                                              sl = row.Close * (1 - stop_loss),
                                              tp = row.Close * (1 + take_profit)))

        # Open Short Positions
        if True == row.sell_signal:
            cost = row.Close * n_shares * COM

            if cash > cost:
                cash -= cost
                active_short_positions.append(Operation
                                              (price = row.Close,
                                               n_shares = n_shares,
                                               sl = row.Close * (1 + stop_loss),
                                               tp = row.Close * (1 -  take_profit)))

        # Value Portfolio for each row
        portfolio_val = 0
        portfolio_val += cash

        ## Value Long positions
        for pos in active_long_positions.copy():
            portfolio_val += row.Close * pos.n_shares

        ## Value Short Positions
        for pos in active_short_positions.copy():
            portfolio_val += (pos.price * n_shares) - (row.Close * n_shares)

        # Add portfolio value to historic
        portfolio_historic.append(portfolio_val)

    last_close = data["Close"].iloc[-1]

    ## Close ALL Long Positions
    for pos in active_long_positions.copy():
        pnl = (pos.price - last_close) * n_shares
        commision = last_close * n_shares * COM
        cash += pnl - commision
        active_short_positions.remove(pos)

    ## Close ALL Short Positions
    for pos in active_short_positions.copy():
        cash += (pos.price * n_shares) - (last_close * n_shares * (1 + COM))
        active_short_positions.remove(pos)

    portfolio_val = cash
    portfolio_historic.append(portfolio_val)

    return portfolio_historic



