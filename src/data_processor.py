from IPython.display import display
import pandas as pd
import numpy as np
from hmmlearn import hmm

def add_technical_indicators(df):
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)

    # ADX (Average Directional Index) - Simplifié
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    tr = pd.concat([df['High']-df['Low'],
                    (df['High']-df['Close'].shift()).abs(), 
                    (df['Low']-df['Close'].shift()).abs()], 
                    axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=14).mean()

    # Volatilité et Returns
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=24).std()
    
    return df.dropna()

def extract_regimes(df, n_regimes=3):
    X = df[['Returns', 'Volatility']].values
    model = hmm.GaussianHMM(
        n_components=n_regimes, 
        covariance_type="full", 
        n_iter=500,           
        tol=0.01,             
        random_state=42
    )
    model.fit(X)

    df = df.copy()
    df['Regime'] = model.predict(X)
    return df, model.transmat_