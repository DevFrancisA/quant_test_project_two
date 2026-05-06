import numpy as np
import math

def add_returns(h_df):
    s = h_df["Close"]
    s_shifted = s.shift(periods=1)
    h_df["Daily Return"] = (s - s_shifted) / s_shifted

    s_shifted_five = s.shift(periods=5)
    h_df["5-day Lagged Return"] = (s - s_shifted_five) / s_shifted_five

    s_shifted_ten = s.shift(periods=10)
    h_df["10-day Lagged Return"] = (s - s_shifted_ten) / s_shifted_ten

def add_SMAs(h_df):
    # simple moving averages
    h = h_df["Close"]
    h_df["10day_SMA"] = (h.rolling(10)).mean()
    h_df["30day_SMA"] = (h.rolling(30)).mean()
    h_df["MA ratio"] = h_df["10day_SMA"] / h_df["30day_SMA"]

def add_rolling_volatility(h_df):
    s = h_df["Daily Return"]
    h_df["rolling_vol"] = (s.rolling(10)).std()

def add_volume(h_df):
    v = h_df["Volume"]
    h_df["10day Vol"] = (v.rolling(10)).mean()
    h_df["vol ratio"] = v / h_df["10day Vol"]

def add_target_variable(h_df):
    r = h_df["Daily Return"]
    h_df["Target"] = np.where(r.shift(periods=-1) > 0, 1, 0)

def add_strategy_return(test_set, pred):
    position = np.where(pred > 0.6, pred, 0)
    test_set["Strategy return"] = test_set["Daily Return"] * position

def add_cumulative_returns(test_set):
    test_set["BH Cumulative Return"] = np.cumprod(test_set["Daily Return"] + 1)
    test_set["Strategy Cumulative Return"] = np.cumprod(test_set["Strategy return"] + 1)

def get_evaluation_metrics(ticker, X_test_returns, y_test, y_pred):
    from sklearn.metrics import accuracy_score

    T = X_test_returns.index.get_level_values(0)[0]
    s_c_r = X_test_returns["Strategy Cumulative Return"]
    bh_x_r = X_test_returns["BH Cumulative Return"]
    d = X_test_returns["Daily Return"]
    a_v = d.std() * math.sqrt(252)
    a_r = d.mean() * 252
    y_pred_binary = (y_pred > 0.5).astype(int)
    m_a = accuracy_score(y_test, y_pred_binary)
    d_m = y_pred_binary.sum()

    return f"""{T}:
    Strategy Total return: {s_c_r.iloc[-1] - 1}
    BH Total return: {bh_x_r.iloc[-1] - 1}
    Volatility: {a_v}
    Sharpe Ratio: {a_r / a_v}
    Model Accuracy: {m_a}
    Days in Market: {d_m}
            """

def visualize_results(full_df, test_df, counter, axes):
    T = full_df.index.get_level_values(0)[0]
    dates_full = full_df.index.get_level_values("Date")
    dates_test = test_df.index.get_level_values("Date")

    axes[0, counter].plot(dates_full, full_df["Close"], label="close")
    axes[0, counter].plot(dates_full, full_df["10day_SMA"], label="10day_SMA")
    axes[0, counter].plot(dates_full, full_df["30day_SMA"], label="30day_SMA")
    axes[0, counter].set_xlabel("Date")
    axes[0, counter].set_ylabel("Price")
    axes[0, counter].set_title(f"{T} Close + SMAs (Full Period)")
    axes[0, counter].legend()

    axes[1, counter].plot(dates_test, test_df["Strategy Cumulative Return"], label="Strategy")
    axes[1, counter].plot(dates_test, test_df["BH Cumulative Return"], label="Buy & Hold")
    axes[1, counter].set_xlabel("Date")
    axes[1, counter].set_ylabel("Cumulative Return")
    axes[1, counter].set_title(f"{T} Strategy vs B&H (Test Period)")
    axes[1, counter].legend()
    axes[1, counter].xaxis.set_major_formatter(__import__("matplotlib").dates.DateFormatter("%Y-%m"))
    axes[1, counter].xaxis.set_major_locator(__import__("matplotlib").dates.MonthLocator(interval=3))
    axes[1, counter].tick_params(axis="x", rotation=45)



