import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm.notebook import tqdm
import json

# Directory di input: in questa cartella, per ogni ticker c'è una sottocartella
input_dir = '/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/RL INPUT/'

# Directory di output per i file normalizzati
output_dir = '/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/'
os.makedirs(output_dir, exist_ok=True)

# Cerca tutti i file CSV che seguono il pattern: ogni sottocartella contiene un file *_rl_input.csv
csv_files = glob.glob(os.path.join(input_dir, '*', '*_complete_rl_input.csv'))
print(f"Trovati {len(csv_files)} file CSV.")

for csv_path in tqdm(csv_files, desc="Normalizzazione file CSV"):
    # Estrai il nome del ticker (si assume che il file si chiami TICKER_rl_input.csv)
    base_name = os.path.basename(csv_path)
    ticker = base_name.split('_')[0]

    # Carica il CSV
    df = pd.read_csv(csv_path)

    # Converti la colonna 'date' in datetime e ordina per data, se esiste
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # Definisci le colonne da escludere dalla normalizzazione
    columns_to_exclude = [
        'date',             # Data temporale
        'label',            # Etichetta testuale
        'day',              # Giorno della settimana (categorico)
        'week',             # Numero della settimana (categorico)
        'SMA5_Above_SMA20', # Indicatore binario
        'Golden_Cross',     # Indicatore binario
        'Death_Cross',      # Indicatore binario
        'Volume_Spike',     # Indicatore binario
        'Volume_Collapse',  # Indicatore binario
        'GARCH_Vol',
        'tau_param_x',
        'tau_param_y'
    ]
    columns_to_exclude = [col for col in columns_to_exclude if col in df.columns]

    # Seleziona le colonne numeriche da normalizzare (tutte le numeriche non escluse)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    columns_to_normalize = [col for col in numeric_columns if col not in columns_to_exclude]

    # (Opzionale) Verifica valori problematici
    print(f"\nVerifica delle colonne per {ticker}...")
    for col in columns_to_normalize:
        has_inf = np.isinf(df[col]).any()
        has_nan = np.isnan(df[col]).any()
        try:
            max_val = df[col].max()
            min_val = df[col].min()
        except Exception as e:
            max_val, min_val = None, None
        if has_inf or has_nan:
            print(f"  {col}: Inf: {has_inf}, NaN: {has_nan}, Min: {min_val}, Max: {max_val}")

    # Crea una copia del DataFrame per i dati normalizzati
    normalized_df = df.copy()

    # Correggi eventuali valori problematici: sostituisci infiniti con NaN e imputa NaN con la mediana
    print(f"Correzione dei valori problematici per {ticker}...")
    for col in columns_to_normalize:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    # Calcola i parametri di normalizzazione per questo ticker
    norm_params = {}
    norm_params['min'] = df[columns_to_normalize].min().to_dict()
    norm_params['max'] = df[columns_to_normalize].max().to_dict()

    # Salva i parametri di normalizzazione in un file JSON specifico per questo ticker
    json_output_path = os.path.join('/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json', f"{ticker}_norm_params.json")
    with open(json_output_path, 'w') as f:
        json.dump(norm_params, f)
    print(f"Parametri di normalizzazione per {ticker} salvati in: {json_output_path}")

    # Applica il Min-Max scaling su ogni colonna numerica da normalizzare
    print(f"Applicazione della normalizzazione Min-Max per {ticker}...")
    for col in tqdm(columns_to_normalize, desc=f"Normalizzo {ticker}", leave=False):
        try:
            column_data = df[col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_values = scaler.fit_transform(column_data).flatten()
            normalized_df[col] = normalized_values
        except Exception as e:
            print(f"ERRORE nella normalizzazione della colonna {col}: {e}")
            normalized_df[col] = df[col]

    # Crea una sottocartella nell'output se non esiste, usando il ticker come nome
    ticker_output_dir = os.path.join(output_dir, ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)

    # Salva il DataFrame normalizzato in un nuovo CSV nella sottocartella del ticker
    output_csv_path = os.path.join(ticker_output_dir, f"{ticker}_normalized.csv")
    normalized_df.to_csv(output_csv_path, index=False)
    print(f"{ticker}: File normalizzato salvato in: {output_csv_path}")