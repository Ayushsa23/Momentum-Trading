import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import exp
import os

# ---------------------------
# USER PARAMETERS (edit here)
# ---------------------------
N = 2000                      # total observations (sum of n1+n2+n3)
seed = 42
np.random.seed(seed)

# Means of log-returns (set directly)
mu1 = 0.005
mu2 = 0.009

# Volatility choices (std dev of log-returns). Adjust as desired.
sigma1 = 0.001    # asset A daily vol
sigma2 = 0.0015   # asset B daily vol

# regime split (must sum to N)
n1 = 700  # "Training" period to find n_hat
n2 = 600  # "Trading" period 1
n3 = 700  # "Trading" period 2
assert n1 + n2 + n3 == N, "n1+n2+n3 must equal N"

# regime correlations
rho1 = 0.9
rho2 = -0.9
rho3 = 0.9

# price base
price_base = 10.0

# trading parameters
L_ema = 20
k = 1.0 - (1.0 / L_ema)
alpha_weights = np.array([k**i for i in range(L_ema)])
initial_capital = 10000000.0
leverage = 1.0                  # l
USE_FRACTIONAL_SHARES = True    # if True, allow fractional shares

# rolling z-score parameters
rolling_K = 200                 # window to compute rolling mean/std for the spread
z_eps = 1e-9                    # small number to prevent division by zero

# output paths
out_dir = "./sim_output_corrected"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------
# SIMULATE BIVARIATE LOG-RETURNS
# ---------------------------
def cov_from(sd1, sd2, rho):
    cov = rho * sd1 * sd2
    return np.array([[sd1**2, cov],
                     [cov,    sd2**2]])

block1 = np.random.multivariate_normal(mean=[mu1, mu2], cov=cov_from(sigma1, sigma2, rho1), size=n1)
block2 = np.random.multivariate_normal(mean=[mu1, mu2], cov=cov_from(sigma1, sigma2, rho2), size=n2)
block3 = np.random.multivariate_normal(mean=[mu1, mu2], cov=cov_from(sigma1, sigma2, rho3), size=n3)

returns = np.vstack([block1, block2, block3])  # shape (N,2)
rA = returns[:, 0]
rB = returns[:, 1]

# convert log-returns to log-prices and prices (base price at t=0 = price_base)
logP_A = np.log(price_base) + np.cumsum(rA)
logP_B = np.log(price_base) + np.cumsum(rB)
priceA = np.exp(logP_A)
priceB = np.exp(logP_B)

dates = pd.RangeIndex(start=0, stop=N, step=1)
prices = pd.DataFrame({'A': priceA, 'B': priceB}, index=dates)
log_returns = pd.DataFrame({'rA': rA, 'rB': rB}, index=dates)

# ... (Plotting for log-returns is unchanged, omitted for brevity) ...
# plt.show() 

# ---------------------------
# Fit OLS to get n_hat (FIXED: NO LOOKAHEAD BIAS)
# ---------------------------
# We use *only* the first n1 points (training period) to find the hedge ratio
y_train = np.log(prices['A'].iloc[:n1])
X_train = sm.add_constant(np.log(prices['B'].iloc[:n1]))

ols_model = sm.OLS(y_train, X_train).fit()
alpha_hat = float(ols_model.params['const'])
n_hat = float(ols_model.params[1])
print(f"\nEstimated regression (on first n1={n1} points):")
print(f"alpha_hat = {alpha_hat:.6f}, n_hat = {n_hat:.6f}")

# Compute spread using this single n_hat for the *entire* series
spread_est = np.log(prices['A']) - n_hat * np.log(prices['B'])
spread_series = pd.Series(spread_est, index=prices.index)

# ---------------------------
# Prepare rolling mean/std for z-score
# ---------------------------
# This is causal (uses data up to and including current time t)
roll_mean = spread_series.rolling(window=rolling_K, min_periods=1).mean().values
roll_std = spread_series.rolling(window=rolling_K, min_periods=1).std(ddof=0).fillna(0).values
roll_std = np.where(roll_std < z_eps, z_eps, roll_std)

z_series = pd.Series((spread_series.values - roll_mean) / roll_std, index=spread_series.index)

# ---------------------------
# SAVE CSVs for inspection
# ---------------------------
prices.to_csv(os.path.join(out_dir, "simulated_prices.csv"), index_label="time")
z_series.to_frame("zscore").to_csv(os.path.join(out_dir, "spread_zscore.csv"), index_label="time")

# ---------------------------
# TRADING LOOP (CORRECTED)
# ---------------------------
cash = float(initial_capital)
hold_A = 0.0  # Start with zero holdings
hold_B = 0.0  # Start with zero holdings
trade_records = []

# We can start trading after the *training period* (n1)
# or after the first EMA window, whichever is later.
# Let's start trading *after* the training period.
start_idx = n1 
if start_idx < (L_ema - 1):
    start_idx = (L_ema - 1) # Ensure we have enough data for EMA

print(f"Trading from index {start_idx} to {N-1}...")

for t in range(start_idx, N):
    
    # --- 1. Get Signal ---
    # raw spread window (most recent L_ema values)
    raw_window = spread_est[(t - (L_ema - 1)):(t + 1)].values
    
    # aligned rolling mean/std window for the same indices
    means_window = roll_mean[(t - (L_ema - 1)):(t + 1)]
    stds_window = roll_std[(t - (L_ema - 1)):(t + 1)]
    
    # compute z-score window (most recent at the end)
    z_window = (raw_window - means_window) / stds_window
    
    # reverse so S_recent[0] = current z_t
    S_recent = z_window[::-1]
    
    # phi computed on the standardized deviations (EMA of z-score)
    phi_t = float(np.dot(alpha_weights, S_recent))
    phi_bar_t = np.tanh(phi_t)
    
    # --- 2. Calculate Target Positions ---
    # Use static capital for position sizing (as in original)
    E_t = (abs(phi_bar_t) * (initial_capital * leverage)) / (1.0 + n_hat)

    if phi_bar_t > 0: # Upward momentum -> Short the spread
        V_A = -E_t
        V_B = +n_hat * E_t
    elif phi_bar_t < 0: # Downward momentum -> Long the spread
        V_A = +E_t
        V_B = -n_hat * E_t
    else:
        V_A = 0.0
        V_B = 0.0

    current_price_A = prices['A'].iloc[t]
    current_price_B = prices['B'].iloc[t]

    if USE_FRACTIONAL_SHARES:
        qA_target = V_A / current_price_A
        qB_target = V_B / current_price_B
    else:
        qA_target = int(np.round(V_A / current_price_A))
        qB_target = int(np.round(V_B / current_price_B))

    # --- 3. Calculate Trades (FIXED) ---
    # delta = target_position - current_holding
    delta_qA = qA_target - hold_A
    delta_qB = qB_target - hold_B

    # --- 4. Update PNL (FIXED) ---
    # Cash flow from trades (negative = cash out, positive = cash in)
    trade_cash_flow = -(delta_qA * current_price_A) - (delta_qB * current_price_B)
    cash += trade_cash_flow
    
    # Update holdings to new target
    hold_A = qA_target
    hold_B = qB_target
    
    # Mark-to-Market value of current holdings
    mtm_value = (hold_A * current_price_A) + (hold_B * current_price_B)
    
    # Total portfolio value
    total_value = cash + mtm_value
    
    # Cumulative PNL
    cum_pnl = total_value - initial_capital

    trade_records.append({
        't': t,
        'priceA': current_price_A,
        'priceB': current_price_B,
        'phi_bar_t': phi_bar_t,
        'qA_target': qA_target,
        'qB_target': qB_target,
        'delta_qA': delta_qA,
        'delta_qB': delta_qB,
        'hold_A': hold_A,
        'hold_B': hold_B,
        'cash': cash,
        'mtm_value': mtm_value,
        'total_value': total_value,
        'cum_pnl': cum_pnl
    })

trades_df = pd.DataFrame(trade_records)
trades_df.to_csv(os.path.join(out_dir, "momentum_on_spread_trade_log_CORRECTED.csv"), index=False)
print("Trade log saved.")

# ---------------------------
# Plots: Updated to show correct PNL and holdings
# ---------------------------
x = trades_df['t'].values  # x-axis

# 1) Prices A and B (on same chart)
plt.figure(figsize=(12, 4))
plt.plot(prices.index, prices['A'], label='Price A')
plt.plot(prices.index, prices['B'], label='Price B')
plt.title("1) Prices A and B vs time")
plt.xlabel("time")
plt.ylabel("Price")
plt.axvline(start_idx, color='r', linestyle='--', label='Trading Start')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot1_prices.png"))
plt.show()

# 2) z-score of spread
plt.figure(figsize=(12, 4))
plt.plot(z_series.index, z_series, label='Z-Score')
plt.title("2) z-score of spread vs time")
plt.xlabel("time")
plt.ylabel("z-score")
plt.axhline(0, color='k', linestyle='--', linewidth=0.6)
plt.axvline(start_idx, color='r', linestyle='--', label='Trading Start')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot2_zscore.png"))
plt.show()

# 3) phi_bar_t (Signal)
plt.figure(figsize=(12, 4))
plt.plot(x, trades_df['phi_bar_t'])
plt.title("3) Signal (phi_bar_t) vs time")
plt.xlabel("time")
plt.ylabel("phi_bar_t")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot3_phi_bar.png"))
plt.show()

# 4) Holdings
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle('Holdings vs time', fontsize=14)
axs[0].plot(x, trades_df['hold_A'], label='Holdings A')
axs[0].set_ylabel('Shares A')
axs[0].grid(True)
axs[1].plot(x, trades_df['hold_B'], label='Holdings B', color='orange')
axs[1].set_ylabel('Shares B')
axs[1].set_xlabel('time')
axs[1].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(out_dir, "plot4_holdings.png"))
plt.show()

# 5) Corrected Cumulative PNL
plt.figure(figsize=(12, 4))
plt.plot(x, trades_df['cum_pnl'])
plt.title("5) Cumulative PnL vs time (Corrected)")
plt.xlabel("time")
plt.ylabel("cum_pnl")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot5_cum_pnl.png"))
plt.show()

# 6) Total Portfolio Value
plt.figure(figsize=(12, 4))
plt.plot(x, trades_df['total_value'])
plt.title("6) Total Portfolio Value vs time")
plt.xlabel("time")
plt.ylabel("Total Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot6_total_value.png"))
plt.show()

print("Saved all plots separately in:", os.path.abspath(out_dir))