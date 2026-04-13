# Survivorship-Bias-Free Momentum Test

$100K initial | 2000-01-01 to 2026-04-11 | Monthly rebalance | Top-10

## Overall Results

| Strategy           | Final Equity   | CAGR   |   Sharpe | Max Drawdown   | Start      | End        |
|:-------------------|:---------------|:-------|---------:|:---------------|:-----------|:-----------|
| Bias-Free Momentum | $647,991       | 7.38%  |    0.387 | -68.74%        | 2000-01-03 | 2026-04-10 |
| Biased Momentum    | $64,240,548    | 27.91% |    0.91  | -66.75%        | 2000-01-03 | 2026-04-10 |
| SPY Buy & Hold     | $743,656       | 7.94%  |    0.492 | -55.19%        | 2000-01-03 | 2026-04-10 |

## Era Breakdown

| Era                           |   Bias-Free Sharpe | Bias-Free CAGR   |   Biased Sharpe | Biased CAGR   |   SPY Sharpe | SPY CAGR   |
|:------------------------------|-------------------:|:-----------------|----------------:|:--------------|-------------:|:-----------|
| Dot-com crash (2000-2002)     |             -0.211 | -14.66%          |          -0.058 | -11.65%       |       -0.518 | -14.27%    |
| Recovery (2003-2006)          |              0.61  | 13.16%           |           2.02  | 65.39%        |        1.115 | 13.71%     |
| Financial crisis (2007-2009)  |             -0.332 | -14.89%          |          -0.033 | -7.25%        |       -0.044 | -5.62%     |
| Bull run (2010-2019)          |              0.538 | 10.99%           |           1.066 | 26.88%        |        0.922 | 13.26%     |
| COVID + aftermath (2020-2022) |              0.413 | 8.77%            |           0.694 | 21.79%        |        0.407 | 7.30%      |
| Recent (2023-2026)            |              1.135 | 36.67%           |           1.815 | 88.33%        |        1.327 | 20.95%     |

## Methodology

- **Bias-Free**: At each rebalance, uses the ACTUAL SP500 constituent list from that date (fja05680/sp500 dataset). Includes companies that later went bankrupt (Enron, Lehman), were acquired, or were delisted.
- **Biased**: Uses today's SP500 members for all historical dates. This is the standard (flawed) approach most backtests use.
- **SPY B&H**: Buy $100K of SPY on day 1, hold forever.

Data source: 737 tickers with price data out of 1081 unique historical constituents.
