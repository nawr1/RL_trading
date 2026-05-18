import pandas as pd
import urllib.request
import zipfile
import os
from datetime import datetime

os.makedirs('raw_data', exist_ok=True)

COLUMNS = ['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
           'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
           'Taker_Buy_Base', 'Taker_Buy_Quote', 'Ignore']

now = datetime.now()
mois = []
for year in range(2019, now.year + 1):
    for month in range(1, 13):
        if year == now.year and month >= now.month:
            break
        mois.append(f"{year:04d}-{month:02d}")

print(f"Periode : {mois[0]} -> {mois[-1]} ({len(mois)} mois)\n")

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

def detect_unit_and_convert(series):
    ts = pd.to_numeric(series, errors='coerce')
    unit = 'us' if ts.median() > 1e15 else 'ms'
    return pd.to_datetime(ts, unit=unit), unit

all_data = []
downloaded_count = 0
skipped = []

for m in mois:
    url      = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip"
    zip_file = f"BTCUSDT-1h-{m}.zip"

    print(f"[{m}]", end=" ", flush=True)

    try:
        urllib.request.urlretrieve(url, zip_file)

        if not zipfile.is_zipfile(zip_file):
            raise ValueError("Fichier invalide (probable 403/404)")

        with zipfile.ZipFile(zip_file, 'r') as z:
            csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None)

        try:
            pd.to_numeric(df.iloc[0, 0])
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

        df.columns = COLUMNS
        df['Open_Time'], unit = detect_unit_and_convert(df['Open_Time'])

        df = df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        df.to_csv(f'raw_data/BTCUSDT-1h-{m}.csv', index=False)
        all_data.append(df)
        downloaded_count += 1
        print(f"OK ({len(df)} lignes) [unit={unit}]")

    except Exception as e:
        print(f"Erreur : {e}")
        skipped.append(m)

    finally:
        if os.path.exists(zip_file):
            os.remove(zip_file)

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values('Open_Time').reset_index(drop=True)

    filename = "BTCUSDT_1h_2019_2026.csv"
    final_df.to_csv(filename, index=False)

    print(f"\nTermine : {downloaded_count}/{len(mois)} mois, {len(final_df):,} lignes")
    print(f"Periode : {final_df['Open_Time'].min()} -> {final_df['Open_Time'].max()}")
    if skipped:
        print(f"Ignores : {', '.join(skipped)}")
else:
    print("Aucune donnee recuperee.")