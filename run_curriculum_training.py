import os
import torch
import numpy as np
from agent import Agent
from env import Environment
from models import Actor
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configurazione
ticker = "ARKG"  # Ticker da utilizzare
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'
csv_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/{ticker}/{ticker}_normalized.csv'
output_dir = f'results/{ticker}_curriculum'

# Crea directory di output se non esiste
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/weights', exist_ok=True)
os.makedirs(f'{output_dir}/test', exist_ok=True)
os.makedirs(f'{output_dir}/analysis', exist_ok=True)

# Verifica esistenza dei file necessari
if not os.path.exists(norm_params_path):
    raise FileNotFoundError(f"File dei parametri di normalizzazione non trovato: {norm_params_path}")

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

# Curriculum learning: definizione dei livelli di commissioni
# Valori realistici finali
realistic_free_trades = 10
realistic_commission_rate = 0.0025  # 0.25%
realistic_min_commission = 1.0

# Valori iniziali ridotti
initial_free_trades = 100
initial_commission_rate = 0.0001  # 0.01%
initial_min_commission = 0.05

# Parametri di training
total_episodes = 200
stages = 5  # Numero di stadi nel curriculum
episodes_per_stage = total_episodes // stages
save_freq = 10
learn_freq = 20

# Parametri di base dell'ambiente (esclusi quelli che verranno aggiornati nel curriculum)
base_env_params = {
    'sigma': 0.1,
    'theta': 0.1,
    'T': len(df_train) - 1,
    'lambd': 0.05,        # Peso penalità posizione ridotto
    'psi': 0.2,           # Peso costi di trading ridotto
    'cost': "trade_l1",
    'max_pos': 4,         # Posizione massima aumentata
    'squared_risk': False,
    'penalty': "tanh",
    'alpha': 3,           # Parametro penalità ridotto
    'beta': 3,            # Parametro penalità ridotto
    'clip': True,
    'scale_reward': 5,    # Fattore di scala ricompense ridotto
    'df': df_train,
    'norm_params_path': norm_params_path,
    'norm_columns': norm_columns,
    'max_step': max_steps
}

# Inizializza l'agente
print("Inizializzazione dell'agente DDPG...")
agent = Agent(
    memory_type="prioritized",
    batch_size=256,         # Dimensione batch aumentata
    max_step=max_steps,
    theta=0.1,              # Parametro rumore OU
    sigma=0.3               # Parametro rumore OU aumentato
)

# Parametri di base per l'addestramento (esclusi quelli specifici del curriculum)
base_train_params = {
    'tau_actor': 0.01,          # Tasso di update soft dell'actor
    'tau_critic': 0.05,         # Tasso di update soft del critic
    'lr_actor': 1e-5,           # Learning rate dell'actor
    'lr_critic': 2e-4,          # Learning rate del critic
    'weight_decay_actor': 1e-6, # Regolarizzazione L2 per l'actor
    'weight_decay_critic': 2e-5, # Regolarizzazione L2 per il critic
    'total_steps': 2000,        # Passi di pretraining
    'weights': f'{output_dir}/weights/',
    'fc1_units_actor': 128,     # Nodi primo layer actor
    'fc2_units_actor': 64,      # Nodi secondo layer actor
    'fc1_units_critic': 256,    # Nodi primo layer critic
    'fc2_units_critic': 128,    # Nodi secondo layer critic
    'learn_freq': learn_freq,
    'decay_rate': 1e-6,         # Decay rate per esplorazione
    'explore_stop': 0.1,        # Probabilità minima di esplorazione
    'tensordir': f'{output_dir}/runs/',
    'progress': "tqdm",         # Mostra barra di avanzamento
}

# Funzione per calcolare i parametri di un dato stadio del curriculum
def get_curriculum_params(stage, total_stages):
    """Calcola i parametri di commissione per un dato stadio del curriculum."""
    # Calcola il fattore di progresso (da 0 a 1)
    progress = stage / (total_stages - 1) if total_stages > 1 else 1
    
    # Calcola i parametri interpolando linearmente tra i valori iniziali e finali
    free_trades = int(initial_free_trades + (realistic_free_trades - initial_free_trades) * progress)
    commission_rate = initial_commission_rate + (realistic_commission_rate - initial_commission_rate) * progress
    min_commission = initial_min_commission + (realistic_min_commission - initial_min_commission) * progress
    
    return {
        'free_trades_per_month': free_trades,
        'commission_rate': commission_rate,
        'min_commission': min_commission
    }

# Avvia il training con curriculum
print(f"Avvio del training curriculum per {ticker} - {total_episodes} episodi totali, {stages} stadi...")

# Salva dettagli del curriculum per riferimento
curriculum_details = []

for stage in range(stages):
    # Calcola i parametri per questo stadio
    curriculum_params = get_curriculum_params(stage, stages)
    
    # Calcola il numero di episodi per questo stadio
    start_episode = stage * episodes_per_stage
    end_episode = (stage + 1) * episodes_per_stage if stage < stages - 1 else total_episodes
    stage_episodes = end_episode - start_episode
    
    # Aggiorna i parametri dell'ambiente
    env_params = {**base_env_params, **curriculum_params}
    
    # Crea e inizializza l'ambiente
    env = Environment(**env_params)
    
    # Aggiorna il parametro freq per il salvataggio
    train_params = {**base_train_params, 'freq': save_freq}
    
    # Registra i dettagli di questo stadio
    curriculum_details.append({
        'stage': stage + 1,
        'episodes': (start_episode, end_episode - 1),
        'free_trades_per_month': curriculum_params['free_trades_per_month'],
        'commission_rate': curriculum_params['commission_rate'],
        'min_commission': curriculum_params['min_commission']
    })
    
    print(f"\nStadio {stage + 1}/{stages} (Episodi {start_episode}-{end_episode-1}):")
    print(f"  Operazioni gratuite al mese: {curriculum_params['free_trades_per_month']}")
    print(f"  Commissione percentuale: {curriculum_params['commission_rate']*100:.5f}%")
    print(f"  Commissione minima: {curriculum_params['min_commission']:.2f}€")
    
    # Avvia il training per questo stadio
    agent.train(
        env=env,
        total_episodes=stage_episodes,
        **train_params
    )
    
    print(f"Completato stadio {stage + 1}/{stages}.")

# Salva i dettagli del curriculum
curriculum_df = pd.DataFrame(curriculum_details)
curriculum_df.to_csv(f"{output_dir}/curriculum_details.csv", index=False)
print(f"Dettagli del curriculum salvati in: {output_dir}/curriculum_details.csv")

print(f"Training curriculum completato per {ticker}!")
print(f"I modelli addestrati sono stati salvati in: {output_dir}/weights/")
print(f"I log per TensorBoard sono stati salvati in: {output_dir}/runs/")

# Valutazione finale sul dataset di test
print("\nAvvio della valutazione finale sul dataset di test...")

# Funzione per valutare un modello
def evaluate_model(model_file, env, agent):
    """Valuta le performance di un modello sul dataset di test."""
    model_path = os.path.join(f'{output_dir}/weights/', model_file)
    
    # Carica il modello
    model = Actor(env.state_size, fc1_units=128, fc2_units=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Assegna il modello all'agente
    agent.actor_local = model
    
    # Resetta l'ambiente all'inizio del test
    env.reset()
    state = env.get_state()
    done = env.done

    # Registra le azioni, posizioni e ricompense
    positions = [0]  # Inizia con posizione 0
    actions = []
    rewards = []
    
    # Esegui un singolo episodio attraverso tutti i dati di test
    while not done:
        action = agent.act(state, noise=False)  # Nessun rumore durante il test
        actions.append(action)
        reward = env.step(action)
        rewards.append(reward)
        state = env.get_state()
        positions.append(env.pi)
        done = env.done

    # Calcola metriche di performance
    cumulative_reward = np.sum(rewards)
    
    if len(rewards) > 1:
        sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)  # Annualizzato
        cum_rewards = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cum_rewards)
        drawdowns = cum_rewards - running_max
        max_drawdown = np.min(drawdowns)
    else:
        sharpe = 0
        max_drawdown = 0

    return {
        'cumulative_reward': cumulative_reward,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': sum(1 for a in actions if abs(a) > 1e-6)
    }

# Inizializza l'ambiente di test con parametri realistici
test_env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_test) - 1,
    lambd=0.05,            # Valore ridotto
    psi=0.2,               # Valore ridotto
    cost="trade_l1",
    max_pos=4,             # Valore aumentato
    squared_risk=False,
    penalty="tanh",
    alpha=3,               # Valore ridotto
    beta=3,                # Valore ridotto
    clip=True,
    scale_reward=5,        # Valore ridotto
    df=df_test,            # Usa i dati di test
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=len(df_test), # Usa tutto il dataset di test
    # Parametri realistici di commissioni per il test
    free_trades_per_month=realistic_free_trades,
    commission_rate=realistic_commission_rate,
    min_commission=realistic_min_commission
)

# Ottieni la lista dei modelli salvati
weights_dir = f'{output_dir}/weights'
model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
model_files.sort(key=lambda x: int(x[4:-4]) if x[4:-4].isdigit() else 0)  # Ordina per numero episodio

if not model_files:
    print("Nessun modello trovato per la valutazione.")
else:
    # Valuta gli ultimi modelli di ogni stadio
    evaluation_results = []
    
    # Trova l'ultimo modello di ogni stadio
    stage_end_episodes = [min((s+1)*episodes_per_stage-1, total_episodes-1) for s in range(stages)]
    
    for stage, end_episode in enumerate(stage_end_episodes):
        # Trova il modello più vicino alla fine dello stadio
        stage_models = [f for f in model_files if int(f[4:-4]) <= end_episode]
        if stage_models:
            stage_model = max(stage_models, key=lambda x: int(x[4:-4]))
            print(f"Valutazione del modello {stage_model} (Stadio {stage+1})...")
            results = evaluate_model(stage_model, test_env, agent)
            results['model'] = stage_model
            results['stage'] = stage + 1
            evaluation_results.append(results)
            
            print(f"  Ricompensa cumulativa: {results['cumulative_reward']:.2f}")
            print(f"  Sharpe ratio: {results['sharpe']:.2f}")
            print(f"  Max drawdown: {results['max_drawdown']:.2f}")
            print(f"  Numero di trade: {results['n_trades']}")
    
    # Salva i risultati
    if evaluation_results:
        eval_df = pd.DataFrame(evaluation_results)
        eval_df.to_csv(f"{output_dir}/test/evaluation_results.csv", index=False)
        print(f"Risultati della valutazione salvati in: {output_dir}/test/evaluation_results.csv")
        
        # Trova il miglior modello
        best_model = max(evaluation_results, key=lambda x: x['cumulative_reward'])
        print(f"\nMiglior modello: {best_model['model']} (Stadio {best_model['stage']})")
        print(f"  Ricompensa cumulativa: {best_model['cumulative_reward']:.2f}")
        print(f"  Sharpe ratio: {best_model['sharpe']:.2f}")
        print(f"  Max drawdown: {best_model['max_drawdown']:.2f}")
        print(f"  Numero di trade: {best_model['n_trades']}")
        
        # Crea un grafico dei risultati della valutazione
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.bar(range(len(evaluation_results)), [r['cumulative_reward'] for r in evaluation_results], color='blue', alpha=0.7)
        plt.xticks(range(len(evaluation_results)), [f"Stage {r['stage']}" for r in evaluation_results])
        plt.title('Ricompensa cumulativa per stadio del curriculum')
        plt.ylabel('Ricompensa cumulativa')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.bar(range(len(evaluation_results)), [r['sharpe'] for r in evaluation_results], color='green', alpha=0.7)
        plt.xticks(range(len(evaluation_results)), [f"Stage {r['stage']}" for r in evaluation_results])
        plt.title('Sharpe ratio per stadio del curriculum')
        plt.ylabel('Sharpe ratio')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test/curriculum_evaluation.png")
        print(f"Grafico dei risultati salvato in: {output_dir}/test/curriculum_evaluation.png")
    else:
        print("Nessun risultato di valutazione disponibile.")