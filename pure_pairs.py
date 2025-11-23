import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS # <-- Import for rolling regression
import os

# ---------------------------
# USER PARAMETERS
# ---------------------------
N = 2000
seed = 42
np.random.seed(seed)

# log-return simulation params
mu1 = 0.005
mu2 = 0.009
sigma1 = 0.001
sigma2 = 0.0015
n1, n2, n3 = 700, 600, 700
rho1, rho2, rho3 = 0.9, -0.9, 0.9
price_base = 10.0

# strategy params (tweak these)
entry_z = 1.5       # enter when z > entry_z or z < -entry_z
exit_band = 0.05    # exit when |z| <= exit_band (close to mean)
exposure = 0.10     # fraction of initial capital used per trade (dollar exposure)
max_hold = 300      # max timesteps to hold a trade (time stop)
stop_loss = 0.25    # stop-loss fraction of exposure (25% adverse move)
tx_cost = 0.0       # <--- SET TO 0.0 AS REQUESTED

# other
initial_capital = 10_000_000.0
USE_FRACTIONAL_SHARES = True

rolling_K = 200     # Rolling window for hedge ratio AND z-score
z_eps = 1e-9

out_dir = "./sim_output"
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

returns = np.vstack([block1, block2, block3])
rA = returns[:,0]; rB = returns[:,1]

# prices
logP_A = np.log(price_base) + np.cumsum(rA)
logP_B = np.log(price_base) + np.cumsum(rB)
priceA = np.exp(logP_A)
priceB = np.exp(logP_B)

dates = pd.RangeIndex(start=0, stop=N, step=1)
prices = pd.DataFrame({'A': priceA, 'B': priceB}, index=dates)

# ---------------------------
# ROLLING hedge ratio & spread & zscore
# (FIXES LOOK-AHEAD BIAS)
# ---------------------------
y = np.log(prices['A'])
X = sm.add_constant(np.log(prices['B']))

# 1. Calculate rolling hedge ratio (n_hat)
# This model regresses y ~ const + B*x
roll_ols_model = RollingOLS(y, X, window=rolling_K)
roll_ols_fit = roll_ols_model.fit()
# Get the 'B' coefficient (the hedge ratio) and back-fill NaNs
n_hat_series = roll_ols_fit.params['B'].bfill()

# 2. Calculate spread using the time-varying hedge ratio
spread_series = np.log(prices['A']) - n_hat_series * np.log(prices['B'])

# 3. Calculate rolling z-score of the spread
#    We use min_periods=rolling_K to wait for a full window (removes unstable early values)
roll_mean = spread_series.rolling(window=rolling_K, min_periods=rolling_K).mean()
roll_std  = spread_series.rolling(window=rolling_K, min_periods=rolling_K).std(ddof=0).fillna(z_eps) # Use fillna(z_eps)
roll_std = np.where(roll_std < z_eps, z_eps, roll_std)
z_series = pd.Series((spread_series - roll_mean) / roll_std, index=spread_series.index)


# ---------------------------
# PAIRS TRADING BACKTEST
# - dollar-neutral entries (V_A = -E, V_B = +E)
# - exit when |z| <= exit_band OR max_hold OR stop_loss breach
# ---------------------------
# Start loop only when all rolling windows are full
start_idx = rolling_K - 1

hold_A = 0.0
hold_B = 0.0
position = 0        # 0=flat, +1 = long A short B, -1 = short A long B
entry_t = None
entry_priceA = entry_priceB = 0.0
entry_exposure = 0.0 # <--- ADDED: To track exposure for stop-loss
cumulative_realized = 0.0

trade_logs = []

for t in range(start_idx, N):
    z_t = float(z_series.iloc[t])
    priceA_t = prices['A'].iloc[t]
    priceB_t = prices['B'].iloc[t]

    trade_cf = 0.0
    qA_target = hold_A
    qB_target = hold_B

    # ENTRY: only when flat
    if position == 0:
        if z_t > entry_z:
            # short A, long B using dollar-neutral exposure E_dollars
            E_dollars = exposure * initial_capital
            entry_exposure = E_dollars # <--- STORE EXPOSURE
            V_A = -E_dollars
            V_B = +E_dollars
            qA_target = V_A / priceA_t
            qB_target = V_B / priceB_t
            position = -1
            entry_t = t
            entry_priceA = priceA_t; entry_priceB = priceB_t
        elif z_t < -entry_z:
            # long A, short B
            E_dollars = exposure * initial_capital
            entry_exposure = E_dollars # <--- STORE EXPOSURE
            V_A = +E_dollars
            V_B = -E_dollars
            qA_target = V_A / priceA_t
            qB_target = V_B / priceB_t
            position = +1
            entry_t = t
            entry_priceA = priceA_t; entry_priceB = priceB_t

    else:
        # IN POSITION: check exit conditions
        held_for = t - entry_t if entry_t is not None else 0
        exit_flag = False
        reason = ''

        # --- STOP-LOSS LOGIC (IMPLEMENTED) ---
        # Calculate unrealized PnL *for this trade*
        unrealized_pnl = hold_A * (priceA_t - entry_priceA) + hold_B * (priceB_t - entry_priceB)
        
        # Check if PnL has dropped below the stop_loss fraction of initial exposure
        if unrealized_pnl < (-stop_loss * entry_exposure):
            exit_flag = True
            reason = 'stop_loss'
        # -------------------------------------
        elif abs(z_t) <= exit_band: # exit if mean crossed
            exit_flag = True
            reason = 'z_returned'
        elif held_for >= max_hold: # exit if max_hold reached
            exit_flag = True
            reason = 'time_stop'

        if exit_flag:
            qA_target = 0.0
            qB_target = 0.0
            position = 0
            entry_t = None
            entry_exposure = 0.0 # <--- RESET EXPOSURE

    # compute deltas and execution cashflows
    delta_qA = qA_target - hold_A
    delta_qB = qB_target - hold_B

    # cashflow: positive when selling
    if delta_qA > 0:
        amount_bought_a = delta_qA * priceA_t * (1.0 + tx_cost) # tx_cost is 0.0
        amount_sold_a = 0.0
    else:
        amount_bought_a = 0.0
        amount_sold_a = -delta_qA * priceA_t * (1.0 - tx_cost) # tx_cost is 0.0

    if delta_qB > 0:
        amount_bought_b = delta_qB * priceB_t * (1.0 + tx_cost) # tx_cost is 0.0
        amount_sold_b = 0.0
    else:
        amount_bought_b = 0.0
        amount_sold_b = -delta_qB * priceB_t * (1.0 - tx_cost) # tx_cost is 0.0

    trade_cf = (amount_sold_a - amount_bought_a) + (amount_sold_b - amount_bought_b)
    cumulative_realized += trade_cf

    # update holdings
    hold_A += delta_qA
    hold_B += delta_qB

    # compute MTM
    hold_value = hold_A * priceA_t + hold_B * priceB_t
    total_pnl = cumulative_realized + hold_value

    # record
    trade_logs.append({
        't': t,
        'z': z_t,
        # 'phi_t' and 'phi_bar_t' removed as requested
        'position': position,
        'qA': hold_A,
        'qB': hold_B,
        'delta_qA': delta_qA,
        'delta_qB': delta_qB,
        'amount_bought_a': amount_bought_a,
        'amount_sold_a': amount_sold_a,
        'amount_bought_b': amount_bought_b,
        'amount_sold_b': amount_sold_b,
        'trade_cf': trade_cf,
        'realized_cum_pnl': cumulative_realized,
        'hold_value': hold_value,
        'total_pnl': total_pnl,
        'priceA': priceA_t,
        'priceB': priceB_t
    })

trades_df = pd.DataFrame(trade_logs)
trades_df.to_csv(os.path.join(out_dir, "pairs_trades_corrected.csv"), index=False)

# ---------------------------
# PERFORMANCE SUMMARY
# ---------------------------
# (This logic remains the same, but it's now analyzing the corrected trades)
events = []
cur_pos = 0
entry_row = None
for i, row in trades_df.iterrows():
    pos = row['position']
    if cur_pos == 0 and pos != 0:
        # entry
        entry_row = row
        cur_pos = pos
    elif cur_pos != 0 and pos == 0 and entry_row is not None:
        # exit
        exit_row = row
        mask = (trades_df['t'] >= entry_row['t']) & (trades_df['t'] <= exit_row['t'])
        realized_trade = trades_df.loc[mask, 'trade_cf'].sum()
        events.append({
            'entry_t': int(entry_row['t']),
            'exit_t': int(exit_row['t']),
            'entry_z': entry_row['z'],
            'exit_z': exit_row['z'],
            'realized_pnl': realized_trade,
            'duration': int(exit_row['t'] - entry_row['t'])
        })
        entry_row = None
        cur_pos = 0

events_df = pd.DataFrame(events)
num_trades = len(events_df)
wins = (events_df['realized_pnl'] > 0).sum() if num_trades>0 else 0
total_realized = trades_df['realized_cum_pnl'].iloc[-1] if len(trades_df)>0 else 0.0
total_total = trades_df['total_pnl'].iloc[-1] if len(trades_df)>0 else 0.0

print("=== Performance summary (Corrected) ===")
print(f"Rolling Window (K): {rolling_K}, Start Index: {start_idx}")
print(f"Trades: {num_trades}, Wins: {wins}, Win rate: {(wins/num_trades*100) if num_trades>0 else 0:.1f}%")
print(f"Total realized PnL: {total_realized:,.2f}")
print(f"Total (realized+unrealized) PnL: {total_total:,.2f}")
if num_trades>0:
    print(f"Avg trade PnL: {events_df['realized_pnl'].mean():.2f}, median: {events_df['realized_pnl'].median():.2f}")
    print(f"Avg duration: {events_df['duration'].mean():.1f} steps")

# ---------------------------
# PLOTS: z-score and cumulative total PnL
# ---------------------------
plt.figure(figsize=(12,4))
# Plot z-score only from the start_idx where trading begins
plt.plot(z_series.index[start_idx:], z_series.values[start_idx:], label='z-score')
plt.axhline(entry_z, color='r', linestyle='--', label=f'entry +{entry_z}')
plt.axhline(-entry_z, color='r', linestyle='--', label=f'entry -{entry_z}')
plt.axhline(0, color='k', linewidth=0.6)
plt.title(f"Z-score of spread (from t={start_idx})")
plt.xlabel("time")
plt.ylabel("z-score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pairs_zscore_corrected.png"))
plt.show()

plt.figure(figsize=(12,4))
#plt.plot(trades_df['t'], trades_df['total_pnl'], label='total_pnl (realized + unrealized)')
plt.plot(trades_df['t'], trades_df['realized_cum_pnl'], alpha=0.7)
plt.title("Cumulative PnL(pairs strategy)")
plt.xlabel("time")
plt.ylabel("PnL (currency)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pairs_total_pnl_corrected.png"))
plt.show()

print("Saved outputs & plots to:", os.path.abspath(out_dir))