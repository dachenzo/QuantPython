import fastfactor

# prices = [100, 105, 110, 120, 125, 130, 140]

# lookback = 3

# momentum_vals = fastfactor.momentum(prices, lookback)

# print(momentum_vals)

prices = [100.0, 102.0, 98.0, 105.0, 110.0, 120.0]
returns = fastfactor.log_returns(prices)
volatility = fastfactor.rolling_volatility(returns, 3)

print("Log Returns:", returns)
print("Rolling Volatility:", volatility)