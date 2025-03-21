import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from agent import Agent
from env import Environment
from models import Actor

# Configurazione
ticker = "ARKG"
output_dir = f'results/{ticker}'
test_file = f'{output_dir}/test/{ticker}_test.csv'
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'

# Verifica che i file necessari esistano
if not os.path.exists(test_file):
    raise FileNotFoundError(f"File di test non trovato: {test_file}")

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

# Carica il dataset di test
df_test = pd.read_csv(test_file)
print(f"Dataset di test caricato: {len(df_test)} righe")

if 'date' in df_test.columns:
    df_test['date'] = pd.to_datetime(df_test['date'])
    print(f"Periodo di test: {df_test['date'].min()} - {df_test['date'].max()}")

# Ottieni la lista dei modelli salvati
weights_dir = f'{output_dir}/weights'
model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
model_files.sort(key=lambda x: int(x[4:-4]) if x[4:-4].isdigit() else 0)  # Ordina per numero episodio

if not model_files:
    raise FileNotFoundError(f"Nessun modello trovato in {weights_dir}")

print(f"Trovati {len(model_files)} modelli addestrati")

# Funzione per valutare un singolo modello
def evaluate_model(model_path, env, agent):
    """Valuta le performance di un modello sul dataset di test."""
    # Carica il modello
    model = Actor(env.state_size, fc1_units=128, fc2_units=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Assegna il modello all'agente
    agent.actor_local = model  # Aggiungi questa riga
    
    # Resetta l'ambiente all'inizio del test
    env.reset()
    state = env.get_state()
    done = env.done

    # Registra le azioni, posizioni e ricompense
    positions = [0]  # Inizia con posizione 0
    actions = []
    rewards = []
    prices = []
    dates = []

    # Esegui un singolo episodio attraverso tutti i dati di test
    while not done:
        action = agent.act(state, noise=False)  # Nessun rumore durante il test
        actions.append(action)
        reward = env.step(action)
        rewards.append(reward)
        state = env.get_state()
        positions.append(env.pi)
        prices.append(env.p)
        if 'date' in df_test.columns and env.current_index < len(df_test):
            dates.append(df_test['date'].iloc[env.current_index])
        done = env.done

    # Calcola metriche di performance
    cumulative_reward = np.sum(rewards)
    # Calcola il P&L direttamente dai prezzi e posizioni
    # Assicurati che gli array abbiano la stessa lunghezza
    if len(prices) > 1:
        # Calcola le differenze di prezzo
        price_changes = np.diff(np.array(prices))
        # Usa le posizioni escludendo la prima e l'ultima
        relevant_positions = np.array(positions[:-1])
        # Assicurati che abbiano la stessa lunghezza
        if len(price_changes) == len(relevant_positions):
            pnl = np.sum(relevant_positions * price_changes)
        else:
            # Se le lunghezze non corrispondono, taglia all'array più corto
            min_length = min(len(price_changes), len(relevant_positions))
            pnl = np.sum(relevant_positions[:min_length] * price_changes[:min_length])
    else:
        pnl = 0
        
    # Calcola altre metriche di performance
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
        'positions': positions,
        'actions': actions,
        'rewards': rewards,
        'prices': prices,
        'dates': dates,
        'cumulative_reward': cumulative_reward,
        'pnl': pnl,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

# Inizializza l'ambiente con i dati di test
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

# Inizializza l'agente
agent = Agent()

# Valuta tutti i modelli o solo alcuni specifici
models_to_evaluate = model_files  # Tutti i modelli
# Oppure seleziona solo alcuni modelli specifici
# models_to_evaluate = ['ddpg10.pth', 'ddpg50.pth', 'ddpg90.pth']

# Dizionario per memorizzare i risultati
results = {}

# Esegui la valutazione per ogni modello
for model_file in models_to_evaluate:
    model_path = os.path.join(weights_dir, model_file)
    print(f"Valutazione del modello {model_file}...")
    
    results[model_file] = evaluate_model(model_path, env, agent)
    
    # Stampa i risultati parziali
    print(f"  Ricompensa cumulativa: {results[model_file]['cumulative_reward']:.2f}")
    print(f"  P&L: {results[model_file]['pnl']:.2f}")
    print(f"  Sharpe ratio: {results[model_file]['sharpe']:.2f}")
    print(f"  Max drawdown: {results[model_file]['max_drawdown']:.2f}")
    print("")

# Identifica il miglior modello in base alla ricompensa cumulativa
best_model = max(results.items(), key=lambda x: x[1]['cumulative_reward'])[0]
print(f"Miglior modello: {best_model}")
print(f"  Ricompensa cumulativa: {results[best_model]['cumulative_reward']:.2f}")
print(f"  P&L: {results[best_model]['pnl']:.2f}")
print(f"  Sharpe ratio: {results[best_model]['sharpe']:.2f}")

# Crea una visualizzazione per il miglior modello
plt.figure(figsize=(14, 10))

# Ottieni i dati del miglior modello
best_results = results[best_model]
positions = best_results['positions']
rewards = best_results['rewards']
prices = best_results['prices']
dates = best_results['dates'] if best_results['dates'] else range(len(prices))

# Grafico dei prezzi e posizioni
plt.subplot(3, 1, 1)
plt.plot(dates, prices, label='Prezzo', color='blue', alpha=0.7)
plt.title(f"Performance del modello {best_model} su dati di test - {ticker}")
plt.ylabel('Prezzo')
plt.xticks(rotation=45)
plt.legend(loc='upper left')

# Aggiungi posizioni su un asse secondario
ax2 = plt.gca().twinx()
ax2.plot(dates, positions[:-1], label='Posizione', color='red', linestyle='--')
ax2.set_ylabel('Posizione')
ax2.set_ylim(-3, 3)  # Adatta questo range in base alle posizioni effettive
ax2.legend(loc='upper right')

# Grafico delle azioni (trades)
plt.subplot(3, 1, 2)
plt.bar(dates, best_results['actions'], label='Azioni', color='green', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.ylabel('Azione (trade)')
plt.legend()

# Grafico del rendimento cumulativo
plt.subplot(3, 1, 3)
cum_rewards = np.cumsum(rewards)
plt.plot(dates, cum_rewards, label='Ricompensa cumulativa', color='purple')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Data')
plt.ylabel('Ricompensa cumulativa')
plt.legend()

# Salva il grafico
plt.tight_layout()
plt.savefig(f'{output_dir}/test/best_model_performance.png')
plt.show()

# Crea un grafico di confronto per tutti i modelli
plt.figure(figsize=(14, 8))

# Confronto delle ricompense cumulative per i diversi modelli
for model_file, model_results in results.items():
    episode = int(model_file[4:-4]) if model_file[4:-4].isdigit() else 0
    cum_rewards = np.cumsum(model_results['rewards'])
    plt.plot(range(len(cum_rewards)), cum_rewards, label=f'Episodio {episode}')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Timestep')
plt.ylabel('Ricompensa cumulativa')
plt.title(f'Confronto delle performance dei modelli - {ticker}')
plt.legend()
plt.grid(True, alpha=0.3)

# Salva il grafico di confronto
plt.tight_layout()
plt.savefig(f'{output_dir}/test/models_comparison.png')
plt.show()

print(f"Visualizzazioni salvate in: {output_dir}/test/")

# Salva i risultati in un file CSV per analisi future
results_summary = []
for model_file, model_results in results.items():
    episode = int(model_file[4:-4]) if model_file[4:-4].isdigit() else 0
    results_summary.append({
        'model': model_file,
        'episode': episode,
        'cumulative_reward': model_results['cumulative_reward'],
        'pnl': model_results['pnl'],
        'sharpe': model_results['sharpe'],
        'max_drawdown': model_results['max_drawdown']
    })

# Converti in DataFrame e salva
results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('episode')
results_df.to_csv(f'{output_dir}/test/evaluation_results.csv', index=False)
print(f"Risultati salvati in: {output_dir}/test/evaluation_results.csv")