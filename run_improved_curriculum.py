import os
import torch
import numpy as np
from agent import Agent
from env import Environment
from models import Actor
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from utils_denorm import summarize_real_performance, plot_real_performance

# Configurazione
ticker = "ARKG"  # Ticker da utilizzare
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'
csv_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/{ticker}/{ticker}_normalized.csv'
output_dir = f'results/{ticker}_improved_curriculum'

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

# Curriculum learning: definizione dei livelli
# Definizione dei parametri del curriculum
total_episodes = 250
stages = 5  # Numero di stadi nel curriculum
episodes_per_stage = total_episodes // stages
save_freq = 10
learn_freq = 20

# STAGE 1: Posizioni limitate, nessuna commissione, incentivo alla stabilità
stage1_params = {
    'max_pos': 1.0,  # Posizione massima molto limitata
    'free_trades_per_month': 1000,  # Praticamente nessuna commissione
    'commission_rate': 0.0001,
    'min_commission': 0.01,
    'trading_frequency_penalty_factor': 0.2,  # Penalità significativa per trading frequente
    'position_stability_bonus_factor': 0.3,  # Bonus per posizioni stabili
    'lambd': 0.05  # Penalità piccola per posizione
}

# STAGE 2: Posizioni leggermente più ampie, basse commissioni
stage2_params = {
    'max_pos': 2.0,
    'free_trades_per_month': 100,
    'commission_rate': 0.0005,
    'min_commission': 0.1,
    'trading_frequency_penalty_factor': 0.3,
    'position_stability_bonus_factor': 0.25,
    'lambd': 0.05
}

# STAGE 3: Posizioni medie, commissioni moderate
stage3_params = {
    'max_pos': 3.0,
    'free_trades_per_month': 30,
    'commission_rate': 0.001,
    'min_commission': 0.3,
    'trading_frequency_penalty_factor': 0.4,
    'position_stability_bonus_factor': 0.2,
    'lambd': 0.05
}

# STAGE 4: Posizioni ampie, commissioni quasi realistiche
stage4_params = {
    'max_pos': 4.0,
    'free_trades_per_month': 15,
    'commission_rate': 0.002,
    'min_commission': 0.7,
    'trading_frequency_penalty_factor': 0.5,
    'position_stability_bonus_factor': 0.15,
    'lambd': 0.05
}

# STAGE 5: Posizioni complete, commissioni realistiche
stage5_params = {
    'max_pos': 4.0,
    'free_trades_per_month': 10,
    'commission_rate': 0.0025,
    'min_commission': 1.0,
    'trading_frequency_penalty_factor': 0.5,
    'position_stability_bonus_factor': 0.1,
    'lambd': 0.05
}

# Lista dei parametri per ogni stadio
stage_params = [stage1_params, stage2_params, stage3_params, stage4_params, stage5_params]

# Parametri di base dell'ambiente (comuni a tutti gli stadi)
base_env_params = {
    'sigma': 0.1,
    'theta': 0.1,
    'T': len(df_train) - 1,
    'psi': 0.2,
    'cost': "trade_l1",
    'squared_risk': False,
    'penalty': "tanh",
    'alpha': 3,
    'beta': 3,
    'clip': True,
    'scale_reward': 5,
    'df': df_train,
    'norm_params_path': norm_params_path,
    'norm_columns': norm_columns,
    'max_step': max_steps
}

# Inizializza l'agente
print("Inizializzazione dell'agente DDPG...")
agent = Agent(
    memory_type="prioritized",
    batch_size=256,
    max_step=max_steps,
    theta=0.1,
    sigma=0.3
)

# Parametri di base per l'addestramento
base_train_params = {
    'tau_actor': 0.01,
    'tau_critic': 0.05,
    'lr_actor': 1e-5,
    'lr_critic': 2e-4,
    'weight_decay_actor': 1e-6,
    'weight_decay_critic': 2e-5,
    'total_steps': 2000,
    'weights': f'{output_dir}/weights/',
    'fc1_units_actor': 128,
    'fc2_units_actor': 64,
    'fc1_units_critic': 256,
    'fc2_units_critic': 128,
    'learn_freq': learn_freq,
    'decay_rate': 1e-6,
    'explore_stop': 0.1,
    'tensordir': f'{output_dir}/runs/',
    'progress': "tqdm",
}

# Avvia il training con curriculum
print(f"Avvio del training curriculum avanzato per {ticker} - {total_episodes} episodi totali, {stages} stadi...")

# Salva dettagli del curriculum per riferimento
curriculum_details = []

for stage, stage_param in enumerate(stage_params):
    # Calcola il numero di episodi per questo stadio
    start_episode = stage * episodes_per_stage
    end_episode = (stage + 1) * episodes_per_stage if stage < stages - 1 else total_episodes
    stage_episodes = end_episode - start_episode
    
    # Combina i parametri di base con quelli dello stadio
    env_params = {**base_env_params, **stage_param}
    
    # Crea e inizializza l'ambiente
    env = Environment(**env_params)
    
    # Aggiorna il parametro freq per il salvataggio
    train_params = {**base_train_params, 'freq': save_freq}
    
    # Registra i dettagli di questo stadio
    curriculum_details.append({
        'stage': stage + 1,
        'episodes': (start_episode, end_episode - 1),
        'max_pos': stage_param['max_pos'],
        'free_trades_per_month': stage_param['free_trades_per_month'],
        'commission_rate': stage_param['commission_rate'],
        'min_commission': stage_param['min_commission'],
        'trading_frequency_penalty_factor': stage_param['trading_frequency_penalty_factor'],
        'position_stability_bonus_factor': stage_param['position_stability_bonus_factor'],
        'lambd': stage_param['lambd']
    })
    
    print(f"\nStadio {stage + 1}/{stages} (Episodi {start_episode}-{end_episode-1}):")
    print(f"  Posizione massima: {stage_param['max_pos']}")
    print(f"  Operazioni gratuite al mese: {stage_param['free_trades_per_month']}")
    print(f"  Commissione percentuale: {stage_param['commission_rate']*100:.5f}%")
    print(f"  Commissione minima: {stage_param['min_commission']:.2f}€")
    print(f"  Fattore penalità trading frequente: {stage_param['trading_frequency_penalty_factor']}")
    print(f"  Fattore bonus stabilità: {stage_param['position_stability_bonus_factor']}")
    
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
print("\nAvvio della valutazione finale sul dataset di test con performance reali...")

# Funzione per valutare un modello
def evaluate_model(model_file, env, agent, df_test, norm_params_path):
    """Valuta le performance di un modello sul dataset di test, incluse metriche reali."""
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
    prices = []
    
    # Esegui un singolo episodio attraverso tutti i dati di test
    while not done:
        action = agent.act(state, noise=False)  # Nessun rumore durante il test
        actions.append(action)
        reward = env.step(action)
        rewards.append(reward)
        state = env.get_state()
        positions.append(env.pi)
        prices.append(env.p)
        done = env.done

    # Calcola le performance normalizzate
    norm_results = {
        'cumulative_reward': np.sum(rewards),
        'n_trades': sum(1 for a in actions if abs(a) > 1e-6),
        'avg_position': np.mean(positions),
        'max_position': np.max(positions)
    }
    
    if len(rewards) > 1:
        norm_results['sharpe'] = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)
        cum_rewards = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cum_rewards)
        drawdowns = cum_rewards - running_max
        norm_results['max_drawdown'] = np.min(drawdowns)
    else:
        norm_results['sharpe'] = 0
        norm_results['max_drawdown'] = 0
    
    # Calcola le performance reali (denormalizzate)
    real_results = summarize_real_performance(
        positions, prices, actions, "Log_Close", norm_params_path,
        commission_rate=0.0025, min_commission=1.0
    )
    
    # Crea grafici della performance reale
    fig = plot_real_performance(df_test, positions, prices, "Log_Close", norm_params_path)
    fig.savefig(f"{output_dir}/analysis/{model_file.replace('.pth', '')}_real_performance.png")
    
    return {**norm_results, **real_results}

# Inizializza l'ambiente di test con parametri realistici
test_env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_test) - 1,
    lambd=0.05,
    psi=0.2,
    cost="trade_l1",
    max_pos=4.0,
    squared_risk=False,
    penalty="tanh",
    alpha=3,
    beta=3,
    clip=True,
    scale_reward=5,
    df=df_test,
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=len(df_test),
    free_trades_per_month=10,
    commission_rate=0.0025,
    min_commission=1.0,
    trading_frequency_penalty_factor=0.5,
    position_stability_bonus_factor=0.1
)

# Ottieni la lista dei modelli salvati
weights_dir = f'{output_dir}/weights'
model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
model_files.sort(key=lambda x: int(x[4:-4]) if x[4:-4].isdigit() else 0)

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
            results = evaluate_model(stage_model, test_env, agent, df_test, norm_params_path)
            results['model'] = stage_model
            results['stage'] = stage + 1
            evaluation_results.append(results)
            
            print(f"  Ricompensa cumulativa: {results['cumulative_reward']:.2f}")
            print(f"  P&L reale: ${results['total_pnl']:.2f}")
            print(f"  Costi di trading totali: ${results['total_trading_costs']:.2f}")
            print(f"  Profitto netto: ${results['net_profit']:.2f}")
            print(f"  Sharpe ratio: {results['sharpe']:.2f}")
            print(f"  Numero di trade: {results['n_trades']}")
    
    # Salva i risultati
    if evaluation_results:
        eval_df = pd.DataFrame(evaluation_results)
        eval_df.to_csv(f"{output_dir}/test/evaluation_results.csv", index=False)
        print(f"Risultati della valutazione salvati in: {output_dir}/test/evaluation_results.csv")
        
        # Trova il miglior modello basato sul profitto netto reale
        best_model = max(evaluation_results, key=lambda x: x['net_profit'])
        print(f"\nMiglior modello basato sul profitto netto: {best_model['model']} (Stadio {best_model['stage']})")
        print(f"  Ricompensa cumulativa: {best_model['cumulative_reward']:.2f}")
        print(f"  P&L reale: ${best_model['total_pnl']:.2f}")
        print(f"  Costi di trading totali: ${best_model['total_trading_costs']:.2f}")
        print(f"  Profitto netto: ${best_model['net_profit']:.2f}")
        print(f"  Sharpe ratio: {best_model['sharpe']:.2f}")
        print(f"  Numero di trade: {best_model['n_trades']}")
        print(f"  Rendimento di mercato: {best_model['price_return']:.2f}%")
        
        # Crea grafici di confronto dei risultati
        plt.figure(figsize=(14, 10))
        
        # Grafico del profitto netto per stadio
        plt.subplot(2, 2, 1)
        plt.bar(range(len(evaluation_results)), [r['net_profit'] for r in evaluation_results], color='green')
        plt.xticks(range(len(evaluation_results)), [f"Stage {r['stage']}" for r in evaluation_results])
        plt.title('Profitto netto ($) per stadio')
        plt.ylabel('Profitto netto ($)')
        plt.grid(True, alpha=0.3)
        
        # Grafico dello Sharpe ratio per stadio
        plt.subplot(2, 2, 2)
        plt.bar(range(len(evaluation_results)), [r['sharpe'] for r in evaluation_results], color='blue')
        plt.xticks(range(len(evaluation_results)), [f"Stage {r['stage']}" for r in evaluation_results])
        plt.title('Sharpe ratio per stadio')
        plt.ylabel('Sharpe ratio')
        plt.grid(True, alpha=0.3)
        
        # Grafico del numero di trade per stadio
        plt.subplot(2, 2, 3)
        plt.bar(range(len(evaluation_results)), [r['n_trades'] for r in evaluation_results], color='red')
        plt.xticks(range(len(evaluation_results)), [f"Stage {r['stage']}" for r in evaluation_results])
        plt.title('Numero di operazioni per stadio')
        plt.ylabel('Numero di operazioni')
        plt.grid(True, alpha=0.3)
        
        # Confronto P&L vs Costi
        plt.subplot(2, 2, 4)
        bar_width = 0.35
        index = np.arange(len(evaluation_results))
        plt.bar(index, [r['total_pnl'] for r in evaluation_results], bar_width, color='green', label='P&L totale')
        plt.bar(index + bar_width, [r['total_trading_costs'] for r in evaluation_results], bar_width, color='red', label='Costi totali')
        plt.xticks(index + bar_width/2, [f"Stage {r['stage']}" for r in evaluation_results])
        plt.title('P&L vs Costi di trading')
        plt.ylabel('Valore ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test/curriculum_real_performance.png")
        print(f"Grafico dei risultati salvato in: {output_dir}/test/curriculum_real_performance.png")
        
    else:
        print("Nessun risultato di valutazione disponibile.")