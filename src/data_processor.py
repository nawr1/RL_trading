import pandas as pd
import numpy as np
from hmmlearn import hmm

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI (14)
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']  = df['EMA12'] - df['EMA26']

    # CCI (20)
    tp         = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp     = tp.rolling(20).mean()
    mean_dev   = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI']  = (tp - sma_tp) / (0.015 * mean_dev)

    # ADX (14)
    plus_dm  = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    tr       = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr      = tr.rolling(14).mean()
    plus_di  = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['ADX'] = dx.rolling(14).mean()

    # Returns & Volatility
    df['Returns']    = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(24).std()

    return df.dropna()

def extract_regimes(df: pd.DataFrame, n_regimes: int = 3):
    X = df[['Returns', 'Volatility']].values

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=500,
        tol=0.01,
        random_state=42
    )
    model.fit(X)

    df_out           = df.copy()
    df_out['Regime'] = model.predict(X)
    mean_returns = [
        df_out.loc[df_out['Regime'] == r, 'Returns'].mean()
        for r in range(n_regimes)
    ]
    rank_map = {old: new for new, old in enumerate(np.argsort(mean_returns))}
    df_out['Regime'] = df_out['Regime'].map(rank_map)

    # Rebuild sorted transition matrix
    order    = np.argsort(mean_returns)
    transmat = model.transmat_[np.ix_(order, order)]

    # Regime statistics
    regime_stats = {}
    for r in range(n_regimes):
        mask = df_out['Regime'] == r
        regime_stats[r] = {
            'label':       ['Bearish', 'Neutral', 'Bullish'][r],
            'mean_return': df_out.loc[mask, 'Returns'].mean(),
            'mean_vol':    df_out.loc[mask, 'Volatility'].mean(),
            'count':       mask.sum(),
            'pct':         mask.mean() * 100,
        }
        print(f"  Regime {r} ({regime_stats[r]['label']:7s}): "
              f"μ_ret={regime_stats[r]['mean_return']:.5f}  "
              f"μ_vol={regime_stats[r]['mean_vol']:.5f}  "
              f"n={regime_stats[r]['count']:,} ({regime_stats[r]['pct']:.1f}%)")

    print("\nTransition matrix (Chapman–Kolmogorov A):")
    print(np.round(transmat, 4))

    return df_out, transmat, model, regime_stats

# if __name__ == "__main__":
#     import os
#     DATA_FILE = os.path.join("data", "BTCUSDT_1h_FINAL.csv")

#     df_raw = pd.read_csv(DATA_FILE, parse_dates=['Open_Time'])
#     df_raw = df_raw.rename(columns={'Open_Time': 'Date'}).set_index('Date')

#     print("Adding technical indicators …")
#     df_feat = add_technical_indicators(df_raw)
#     print(f"  Shape after indicators: {df_feat.shape}")

#     print("\nFitting HMM regimes …")
#     df_reg, transmat, model, stats = extract_regimes(df_feat, n_regimes=3)

#     df_reg.to_csv(os.path.join("data", "BTCUSDT_1h_REGIMES.csv"))
#     np.save(os.path.join("data", "transmat.npy"), transmat)
#     print("\nSaved: data/BTCUSDT_1h_REGIMES.csv  +  data/transmat.npy")