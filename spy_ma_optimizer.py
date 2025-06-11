import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set pandas options to display all columns
pd.set_option('display.max_columns', None)


def get_data():
    """
    Get SPY closing prices from Yahoo Finance for the last 10 years.
    """
    print("Downloading SPY data...")
    data = yf.download(tickers=['SPY'], period='10y', interval='1d', auto_adjust=True)
    data = data[['Close']]
    data.rename(columns={'Close': 'spy'}, inplace=True)
    data.dropna(inplace=True)
    return data


def run_backtest(df, short_window, long_window):
    """
    Runs a moving average crossover backtest.
    """
    # Calculate moving averages
    df['short_ma'] = df['spy'].rolling(window=short_window).mean()
    df['long_ma'] = df['spy'].rolling(window=long_window).mean()

    # Create trading signals
    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1

    # Determine positions to avoid lookahead bias
    df['position'] = df['signal'].shift(1)
    df.fillna(0, inplace=True)

    # Calculate daily returns
    if 'spy_daily_return' not in df.columns:
        df['spy_daily_return'] = df['spy'].pct_change()
    df['strategy_daily_return'] = df['position'] * df['spy_daily_return']
    df.fillna(0, inplace=True)

    return df


def calculate_performance_metrics(df, num_years):
    """
    Calculates and returns performance metrics for a single strategy.
    """
    equity_curve = (1 + df['strategy_daily_return']).cumprod()
    total_return = equity_curve.iloc[-1] - 1
    cagr = (equity_curve.iloc[-1]) ** (1 / num_years) - 1 if num_years > 0 else 0
    sharpe_ratio = np.sqrt(252) * df['strategy_daily_return'].mean() / df['strategy_daily_return'].std() if df[
                                                                                                               'strategy_daily_return'].std() != 0 else 0
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return, "CAGR": cagr,
        "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown
    }


def plot_optimized_results(df, best_strategies, worst_strategies):
    """
    Generates and saves a plot comparing the best and worst strategies.
    """
    plt.figure(figsize=(15, 10))

    # Plot Buy and Hold
    df['spy_daily_return'] = df['spy'].pct_change().fillna(0)
    spy_equity_curve = (1 + df['spy_daily_return']).cumprod()
    plt.plot(spy_equity_curve, label='SPY Buy and Hold', color='black', linewidth=2, linestyle='-')

    # Plot Best Strategies
    for i, row in best_strategies.iterrows():
        short, long = int(row['short']), int(row['long'])
        backtest_df = run_backtest(df.copy(), short, long)
        equity_curve = (1 + backtest_df['strategy_daily_return']).cumprod()
        plt.plot(equity_curve, label=f'Best {short}/{long} (CAGR: {row["CAGR"]:.2%})')

    # Plot Worst Strategies
    for i, row in worst_strategies.iterrows():
        short, long = int(row['short']), int(row['long'])
        backtest_df = run_backtest(df.copy(), short, long)
        equity_curve = (1 + backtest_df['strategy_daily_return']).cumprod()
        plt.plot(equity_curve, label=f'Worst {short}/{long} (CAGR: {row["CAGR"]:.2%})', linestyle='--')

    plt.title('Top 5 vs. Bottom 5 MA Crossover Strategies')
    plt.legend(loc='upper left', ncol=2)
    plt.savefig('optimized_strategies_comparison.png')
    plt.close()


if __name__ == '__main__':
    # --- Parameters ---
    NUM_STRATEGIES = 1000
    SHORT_WINDOW_RANGE = (5, 100)
    LONG_WINDOW_RANGE = (20, 250)

    # --- Data ---
    data_df = get_data()
    data_df['spy_daily_return'] = data_df['spy'].pct_change().fillna(0)
    num_years = (data_df.index[-1] - data_df.index[0]).days / 365.25

    # --- Optimization Loop ---
    results = []
    tested_combinations = set()

    print(f"Running {NUM_STRATEGIES} backtests...")
    for _ in tqdm(range(NUM_STRATEGIES), desc="Optimizing Strategies"):
        # Generate a unique combination of windows
        while True:
            short_window = random.randint(*SHORT_WINDOW_RANGE)
            long_window = random.randint(*LONG_WINDOW_RANGE)
            if long_window > short_window + 5 and (short_window, long_window) not in tested_combinations:
                tested_combinations.add((short_window, long_window))
                break

        backtest_df = run_backtest(data_df.copy(), short_window, long_window)
        performance = calculate_performance_metrics(backtest_df, num_years)
        results.append({
            'short': short_window, 'long': long_window,
            'CAGR': performance['CAGR'], 'Sharpe Ratio': performance['Sharpe Ratio'],
            'Max Drawdown': performance['Max Drawdown']
        })

    # --- Analysis ---
    results_df = pd.DataFrame(results)
    best_5 = results_df.nlargest(5, 'CAGR')
    worst_5 = results_df.nsmallest(5, 'CAGR')

    print("\n--- Top 5 Performing Strategies (by CAGR) ---")
    print(best_5)

    print("\n--- Bottom 5 Performing Strategies (by CAGR) ---")
    print(worst_5)

    # --- Visualization ---
    plot_optimized_results(data_df.copy(), best_5, worst_5)
    print("\nPlot saved to: optimized_strategies_comparison.png")

    # --- Buy and Hold Performance for Comparison ---
    spy_equity = (1 + data_df['spy_daily_return']).cumprod()
    spy_cagr = (spy_equity.iloc[-1]) ** (1 / num_years) - 1
    print(f"\nFor comparison, SPY Buy & Hold CAGR: {spy_cagr:.2%}")