import os
import torch
import numpy as np
from agent import Agent
from env import Environment
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configurazione
ticker = "ARKG"  # Ticker da utilizzare
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'
csv_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/{ticker}/{ticker}_normalized.csv'
output_dir = f'results/{ticker}'

# Crea directory di output se non esiste
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/weights', exist_ok=True)
os.makedirs(f'{output_dir}/test', exist_ok=True)  # Aggiungi cartella per i dati di test

# Verifica esistenza dei file necessari
if not os.path.exists(norm_params_path):
    print("File dei parametri di normalizzazione non trovato. Esecuzione create_norm_params.py...")
    os.system("python3 create_norm_params.py")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File CSV dei dati normalizzati non trovato: {csv_path}")

# Definizione delle feature da utilizzare 
norm_columns = [
    "open", "volume", "change", "day", "week", "adjCloseGold", "adjCloseSpy",
    "Credit_Spread", "Log_Close", "m_plus", "m_minus", "drawdown", "drawup",
    "s_plus", "s_minus", "upper_bound", "lower_bound", "avg_duration", "avg_depth",
    "cdar_95", "VIX_Close", "MACD", "MACD_Signal", "MACD_Histogram", "SMA5",
    "SMA10", "SMA15", "SMA20", "SMA25", "SMA30", "SMA36", "RSI5", "RSI14", "RSI20",
    "RSI25", "ADX5", "ADX10", "ADX15", "ADX20", "ADX25", "ADX30", "ADX35",
    "BollingerLower", "BollingerUpper", "WR5", "WR14", "WR20", "WR25",
    "SMA5_SMA20", "SMA5_SMA36", "SMA20_SMA36", "SMA5_Above_SMA20",
    "Golden_Cross", "Death_Cross", "BB_Position", "BB_Width",
    "BB_Upper_Distance", "BB_Lower_Distance", "Volume_SMA20", "Volume_Change_Pct",
    "Volume_1d_Change_Pct", "Volume_Spike", "Volume_Collapse", "GARCH_Vol",
    "pred_lstm", "pred_gru", "pred_blstm", "pred_lstm_direction",
    "pred_gru_direction", "pred_blstm_direction"
]

# Carica il dataset
print(f"Caricamento dati per {ticker}...")
df = pd.read_csv(csv_path)

# Verifica la presenza di tutte le colonne necessarie
missing_cols = [col for col in norm_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")

# Ordina il dataset per data (se presente)
if 'date' in df.columns:
    print("Ordinamento del dataset per data...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    print(f"Intervallo temporale: {df['date'].min()} - {df['date'].max()}")

# Stampa info sul dataset
print(f"Dataset caricato: {len(df)} righe x {len(df.columns)} colonne")

# Separazione in training e test
train_size = int(len(df) * 0.8)  # 80% per training, 20% per test
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

print(f"Divisione dataset: {len(df_train)} righe per training, {len(df_test)} righe per test")

if 'date' in df.columns:
    print(f"Periodo di training: {df_train['date'].min()} - {df_train['date'].max()}")
    print(f"Periodo di test: {df_test['date'].min()} - {df_test['date'].max()}")

# Salva il dataset di test per usi futuri
test_dir = f'{output_dir}/test'
os.makedirs(test_dir, exist_ok=True)
df_test.to_csv(f'{test_dir}/{ticker}_test.csv', index=False)
print(f"Dataset di test salvato in: {test_dir}/{ticker}_test.csv")

# Parametri per l'ambiente
max_steps = min(1000, len(df_train) - 10)  # Limita la lunghezza massima dell'episodio
print(f"Lunghezza massima episodio: {max_steps} timestep")

# Inizializza l'ambiente
# Inizializza l'ambiente con i dati di test
env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_test) - 1,
    lambd=0.05,            # Utilizziamo il valore ridotto come nel training
    psi=0.2,               # Utilizziamo il valore ridotto come nel training
    cost="trade_l1",
    max_pos=4,             # Utilizziamo il valore aumentato come nel training
    squared_risk=False,
    penalty="tanh",
    alpha=3,               # Utilizziamo il valore ridotto come nel training
    beta=3,                # Utilizziamo il valore ridotto come nel training
    clip=True,
    scale_reward=5,        # Utilizziamo il valore ridotto come nel training
    df=df_test,            # Usa i dati di test
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=len(df_test), # Usa tutto il dataset di test
    # Ripristina i parametri realistici di commissioni per il test
    free_trades_per_month=10,       # Torna al valore realistico
    commission_rate=0.0025,         # Torna al valore realistico
    min_commission=1.0               # Torna al valore realistico
)

# Parametri di training
total_episodes = 200      # AUMENTATO ULTERIORMENTE: Numero di episodi (150 -> 200)
learn_freq = 20           # RIDOTTO ULTERIORMENTE: Frequenza di apprendimento (25 -> 20)
save_freq = 10            # Frequenza di salvataggio dei modelli

# Inizializza l'agente
print("Inizializzazione dell'agente DDPG...")
agent = Agent(
    memory_type="prioritized",
    batch_size=256,         # AUMENTATO ULTERIORMENTE: Dimensione batch (128 -> 256)
    max_step=max_steps,
    theta=0.1,              # MODIFICATO: Parametro rumore OU (0.15 -> 0.1)
    sigma=0.3               # AUMENTATO ULTERIORMENTE: Parametro rumore OU (0.2 -> 0.3)
)

# Avvia il training
print(f"Avvio del training per {ticker} - {total_episodes} episodi...")
agent.train(
    env=env,
    total_episodes=total_episodes,
    tau_actor=0.01,          # RIDOTTO ULTERIORMENTE: Tasso di update soft dell'actor (0.05 -> 0.01)
    tau_critic=0.05,         # AUMENTATO ULTERIORMENTE: Tasso di update soft del critic (0.02 -> 0.05)
    lr_actor=1e-5,           # RIDOTTO ULTERIORMENTE: Learning rate dell'actor (5e-5 -> 1e-5)
    lr_critic=2e-4,          # RIDOTTO ULTERIORMENTE: Learning rate del critic (5e-4 -> 2e-4)
    weight_decay_actor=1e-6, # RIDOTTO: Regolarizzazione L2 per l'actor (1e-5 -> 1e-6)
    weight_decay_critic=2e-5, # RIDOTTO: Regolarizzazione L2 per il critic (1e-4 -> 2e-5)
    total_steps=2000,        # AUMENTATO ULTERIORMENTE: Passi di pretraining (1500 -> 2000)
    weights=f'{output_dir}/weights/',
    freq=save_freq,
    fc1_units_actor=128,     # Mantenuto: Nodi primo layer actor
    fc2_units_actor=64,      # Mantenuto: Nodi secondo layer actor
    fc1_units_critic=256,    # Mantenuto: Nodi primo layer critic
    fc2_units_critic=128,    # Mantenuto: Nodi secondo layer critic
    learn_freq=learn_freq,
    decay_rate=1e-6,         # RIDOTTO ULTERIORMENTE: Decay rate per esplorazione (5e-6 -> 1e-6)
    explore_stop=0.1,        # AUMENTATO ULTERIORMENTE: Probabilità minima di esplorazione (0.05 -> 0.1)
    tensordir=f'{output_dir}/runs/',
    progress="tqdm",         # Mostra barra di avanzamento
)

print(f"Training completato per {ticker}!")
print(f"I modelli addestrati sono stati salvati in: {output_dir}/weights/")
print(f"I log per TensorBoard sono stati salvati in: {output_dir}/runs/")