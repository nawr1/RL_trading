import pandas as pd
import urllib.request
import zipfile
import os
from datetime import datetime

# 1. Creation de l'architecture des dossiers
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw_data")
FINAL_FILE = os.path.join(DATA_DIR, "BTCUSDT_1h_FINAL.csv")

os.makedirs(RAW_DIR, exist_ok=True)
print(f"Structure prete : {RAW_DIR}")

# 2. Configuration et parametres
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

# 3. Telechargement et traitement
all_data = []
downloaded_count = 0
skipped = []

for m in mois:
    url      = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip"
    zip_tmp  = f"temp_{m}.zip"

    print(f"[{m}]", end=" ", flush=True)

    try:
        urllib.request.urlretrieve(url, zip_tmp)

        if not zipfile.is_zipfile(zip_tmp):
            raise ValueError("Fichier invalide (404 Binance)")

        with zipfile.ZipFile(zip_tmp, 'r') as z:
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

        monthly_csv_path = os.path.join(RAW_DIR, f"BTCUSDT-1h-{m}.csv")
        df.to_csv(monthly_csv_path, index=False)
        
        all_data.append(df)
        downloaded_count += 1
        print(f"OK ({len(df)} lignes)")

    except Exception as e:
        print(f"Erreur : {e}")
        skipped.append(m)

    finally:
        if os.path.exists(zip_tmp):
            os.remove(zip_tmp)

# 4. Consolidation finale
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values('Open_Time').reset_index(drop=True)

    final_df.to_csv(FINAL_FILE, index=False)

    print(f"\nTermine")
    print(f"Fichiers sauvegardes dans : {RAW_DIR}")
    print(f"Fichier final : {FINAL_FILE}")
    print(f"Total lignes : {len(final_df):,}")
    
    if skipped:
        print(f"Mois ignores : {', '.join(skipped)}")
else:
    print("\nAucune donnee recuperee.")