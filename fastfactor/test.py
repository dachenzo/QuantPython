import fastfactor
import pandas as pd
import numpy as np
import time

# prices = [100, 105, 110, 120, 125, 130, 140]

# lookback = 3

# momentum_vals = fastfactor.momentum(prices, lookback)

# print(momentum_vals)

# prices = [100.0, 102.0, 98.0, 105.0, 110.0, 120.0]
# returns = fastfactor.log_returns(prices)
# volatility = fastfactor.rolling_volatility(returns, 3)

# print("Log Returns:", returns)
# print("Rolling Volatility:", volatility)



# # Generate synthetic data
# np.random.seed(42)
# n = 100_000  # 100,000 data points
# x = np.random.randn(n).tolist()
# y = np.random.randn(n).tolist()
# window = 50

# # Pandas rolling correlation
# df = pd.DataFrame({"x": x, "y": y})

# start = time.time()
# pandas_corr = df["x"].rolling(window).corr(df["y"])
# pandas_time = time.time() - start

# # FastFactor rolling correlation
# start = time.time()
# fast_corr = fastfactor.rolling_correlation(x, y, window)
# fastfactor_time = time.time() - start

# print(f"Pandas time: {pandas_time:.6f} sec")
# print(f"FastFactor time: {fastfactor_time:.6f} sec")


# prices = [100, 102, 98, 105, 110, 120]
# ema = fastfactor.exponential_moving_average(prices, 3, 2.0);

# print("Exponential Moving Average:", ema)

prices = [100, 102, 98, 105, 110, 120]
percentile_90 = fastfactor.rolling_percentile(prices, 2, 90)

print("90th Percentile:", percentile_90)