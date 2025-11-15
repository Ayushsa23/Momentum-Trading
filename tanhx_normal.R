# Load necessary libraries
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

psi_linear <- function(z) unit_normalize(z)


psi_tanh <- function(z) unit_normalize(tanh(z))


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

shaping_functions <- list(
  linear = psi_linear,
  tanh = psi_tanh
)

skewness_results <- data.frame(M = 1:250)
summary_stats <- tibble::tibble(  Strategy = character(),
  Mean_PnL = numeric(),
  Std_Dev_PnL = numeric(),
  Daily_Skewness = numeric()
)


for (signal_name in names(signals)) {
  for (func_name in names(shaping_functions)) {
    strategy_name <- paste(signal_name, func_name, sep = "_")
    phi <- shaping_functions[[func_name]](signals[[signal_name]])
    
    pnl <- dplyr::lag(phi, 1) * U
    pnl <- na.omit(pnl)
    

    skew_curve <- sapply(1:250, function(M) {
      if (length(pnl) < M) return(NA_real_)
      rolling_pnl_sum <- RcppRoll::roll_sum(pnl, n = M, align = "right", fill = NA)
      sample_skewness(rolling_pnl_sum)
    })
    
    skewness_results[[strategy_name]] <- skew_curve
    
    summary_stats <- summary_stats %>%
      add_row(
        Strategy = strategy_name,
        Mean_PnL = mean(pnl, na.rm = TRUE),
        Std_Dev_PnL = sd(pnl, na.rm = TRUE),
        Daily_Skewness = skew_curve[1]
      )
  }
}


sma_window <- 10

skewness_smoothed <- as.data.frame(lapply(skewness_results[-1], function(col) {
  as.numeric(stats::filter(col, rep(1 / sma_window, sma_window), sides = 2))
}))
skewness_smoothed$M <- skewness_results$M


skew_long <- pivot_longer(skewness_smoothed, -M, names_to = "Strategy", values_to = "Skewness") %>%
  mutate(
    Signal = ifelse(grepl("ema1", Strategy), "Single EMA", "Dual EMA"),
    Shaping_Function = gsub("ema1_|ema2_", "", Strategy)
  )

comparison_plot <- ggplot(skew_long, aes(x = M, y = Skewness, color = Shaping_Function, linetype = Signal)) +
  geom_line(na.rm = TRUE, size = 1) +
  labs(
    title = "Term Structure of Skewness: Linear vs. Tanh",
    subtitle = "Comparing a pure linear signal with a capped linear-like signal (tanh).",
    x = "Return Period (M) / Days",
    y = "Smoothed 'Skewness'"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("linear" = "#1b9e77", "tanh" = "#d95f02")) +
  scale_linetype_manual(values = c("Single EMA" = "solid", "Dual EMA" = "dashed")) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey40") +
  ylim(-0.5, 0.5)


print(comparison_plot)

cat("\n--- Strategy Performance Summary: Linear vs. Tanh ---\n")
print(summary_stats, row.names = FALSE)