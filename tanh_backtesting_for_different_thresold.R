
library(RcppRoll)
library(ggplot2)
library(tidyr)
library(dplyr)

set.seed(42)

ema <- function(x, n) {
  alpha <- 2 / (n + 1)
  as.numeric(stats::filter(x, filter = alpha, method = "recursive", sides = 1))
}

unit_normalize <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(x)
  x / s
}

sample_skewness <- function(x) {
  x <- na.omit(x)
  n <- length(x)
  if (n < 3) return(NA_real_)
  s <- sd(x)
  if (is.na(s) || s == 0) return(NA_real_)
  mean((x - mean(x))^3) / s^3
}
n_obs <- 5000
vol_ema_period <- 20

x <- rcauchy(n_obs, location = 0, scale = 1)

sigma_sq_hat <- ema(x^2, n = vol_ema_period)
sigma_hat <- sqrt(sigma_sq_hat)
valid_idx <- which(!is.na(sigma_hat) & is.finite(sigma_hat))
U <- x[valid_idx] / sigma_hat[valid_idx]

ema1_period <- 20
ema2_fast <- 20
ema2_slow <- 40

V_ema1_raw <- ema(U, n = ema1_period)
V_ema2_raw <- ema(U, n = ema2_fast) - ema(U, n = ema2_slow)

V_ema1 <- unit_normalize(V_ema1_raw)
V_ema2 <- unit_normalize(V_ema2_raw)

max_na_period <- max(ema1_period, ema2_slow)
valid2 <- seq_len(length(U)) > max_na_period
U <- U[valid2]
V_ema1 <- V_ema1[valid2]
V_ema2 <- V_ema2[valid2]

signals <- list(ema1 = V_ema1, ema2 = V_ema2)

psi_linear <- function(z) unit_normalize(z)
psi_tanh   <- function(z) unit_normalize(tanh(z))

shaping_functions <- list(
  linear = psi_linear,
  tanh   = psi_tanh
)
thresholds <- seq(0.2, 1.0, by = 0.1)  
results <- tibble::tibble(
  Strategy = character(),
  Shaping = character(),
  Threshold = numeric(),
  Mean_PnL = numeric(),
  SD_PnL = numeric(),
  Daily_Skewness = numeric(),
  Opportunity_Pct = numeric(),  
  Num_Active = integer(),        
  Mean_Abs_Signal_Active = numeric(),
  Mean_PnL_per_Trade = numeric()
)


for (signal_name in names(signals)) {
  for (func_name in names(shaping_functions)) {
    strategy_name <- paste(signal_name, func_name, sep = "_")
    phi_raw <- shaping_functions[[func_name]](signals[[signal_name]])  # original phi (kept)
   
    phi_baseline <- phi_raw
    pnl_baseline <- dplyr::lag(phi_baseline, 1) * U
    pnl_baseline <- na.omit(pnl_baseline)
    
    for (thr in thresholds) {
      phi_thr <- sign(phi_raw) * pmax(abs(phi_raw) - thr, 0)
      pnl <- dplyr::lag(phi_thr, 1) * U
      pnl <- na.omit(pnl)
      active_flags <- !is.na(dplyr::lag(phi_thr, 1)) & (dplyr::lag(phi_thr, 1) != 0)
      active_flags <- active_flags[-1]
      opportunity_pct <- mean(active_flags, na.rm = TRUE)
      num_active <- sum(active_flags, na.rm = TRUE)
      mean_abs_signal_active <- if (num_active > 0) mean(abs(dplyr::lag(phi_thr, 1))[which(active_flags)]) else NA_real_
      mean_pnl_per_trade <- if (num_active > 0) sum(pnl, na.rm = TRUE) / num_active else NA_real_
      daily_skew <- sample_skewness(pnl)
      
      results <- results %>%
        add_row(
          Strategy = signal_name,
          Shaping = func_name,
          Threshold = thr,
          Mean_PnL = mean(pnl, na.rm = TRUE),
          SD_PnL = sd(pnl, na.rm = TRUE),
          Daily_Skewness = daily_skew,
          Opportunity_Pct = opportunity_pct*100,
          Num_Active = num_active,
          Mean_Abs_Signal_Active = mean_abs_signal_active,
          Mean_PnL_per_Trade = mean_pnl_per_trade
        )
    } # end thresholds
  } # end shaping
} # end signals

# ---- Reporting ----
results_summary <- results %>%
  arrange(Strategy, Shaping, Threshold)

print("\n--- Threshold Study Summary (first 50 rows) ---")
print(utils::head(results_summary, 50))

# Save as CSV for further inspection
write.csv(results_summary, "threshold_study_summary.csv", row.names = FALSE)

# ---- Plots: mean PnL vs Threshold and Opportunity% vs Threshold ----
p1 <- ggplot(results_summary, aes(x = Threshold, y = Mean_PnL, color = Shaping, group = interaction(Shaping, Strategy))) +
  geom_line() + geom_point() +
  facet_wrap(~Strategy) +
  labs(title = "Mean PnL vs Threshold (Symmetric dead-zone)",
       subtitle = "Each facet = signal (ema1 / ema2). Lines compare shaping funcs.",
       y = "Mean PnL", x = "Threshold") +
  theme_minimal()

p2 <- ggplot(results_summary, aes(x = Threshold, y = Opportunity_Pct, color = Shaping, group = interaction(Shaping, Strategy))) +
  geom_line() + geom_point() +
  facet_wrap(~Strategy) +
  labs(title = "Opportunity % vs Threshold",
       subtitle = "Opportunity = fraction of days with non-zero position after thresholding",
       y = "Opportunity (%)", x = "Threshold") +
  theme_minimal()

print(p1)
print(p2)

cat("\nSaved full results to: threshold_study_summary.csv\n")
