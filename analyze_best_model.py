import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from agent import Agent
from env import Environment
from models import Actor

# Configurazione
ticker = "ARKG"
output_dir = f'results/{ticker}'
output_dir = f'results/{ticker}'
analysis_dir = f'{output_dir}/analysis'
os.makedirs(analysis_dir, exist_ok=True)  # Crea la directory se non esiste
test_file = f'{output_dir}/test/{ticker}_test.csv'
best_model_file = "ddpg10.pth"  # Il modello che vogliamo analizzare
comparison_model_file = "ddpg60.pth"  # Un modello che ha performance peggiori per confronto
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'

# Verifica che i file necessari esistano
if not os.path.exists(test_file):
    raise FileNotFoundError(f"File di test non trovato: {test_file}")

# Definizione delle feature da utilizzare (stesse del training)
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

# Funzione per valutare e analizzare un modello
def analyze_model(model_file, env, agent):
    """Valuta ed analizza in dettaglio le performance di un modello."""
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
    dates = []
    trade_costs = []  # Per registrare i costi di trading
    position_penalties = []  # Per registrare le penalità sulla posizione
    trend_rewards = []  # Per registrare le ricompense di trend
    pnls = []  # Per registrare i PnL step-by-step

    # Esegui un singolo episodio attraverso tutti i dati di test
    while not done:
        action = agent.act(state, noise=False)  # Nessun rumore durante il test
        actions.append(action)
        
        # Registra lo stato prima dell'azione
        if 'date' in df_test.columns and env.current_index < len(df_test):
            dates.append(df_test['date'].iloc[env.current_index])
        prices.append(env.p)
        
        # Calcola e registra componenti separate della ricompensa
        # Posizione attuale
        pi_prev = env.pi
        # Calcola la nuova posizione dopo l'azione
        pi_next = np.clip(pi_prev + action, -env.max_pos, env.max_pos)
        
        # Costo di trading
        if abs(action) > 1e-6:
            order_value = abs(action) * env.p
            trade_cost = max(order_value * env.commission_rate, env.min_commission)
        else:
            trade_cost = 0
        trade_costs.append(trade_cost)
        
        # Penalità per dimensione della posizione
        position_penalty = env.lambd * pi_next ** 2 * env.squared_risk
        position_penalties.append(position_penalty)
        
        # Esegui l'azione e ottieni la ricompensa totale
        reward = env.step(action)
        rewards.append(reward)
        
        # Calcola il PnL "grezzo" (senza penalità)
        if len(prices) > 1:
            price_change = prices[-1] - prices[-2]
            pnl = pi_prev * price_change
            pnls.append(pnl)
        else:
            pnls.append(0)
        
        # Aggiorna lo stato
        state = env.get_state()
        positions.append(env.pi)
        done = env.done

    # Calcolo di metriche aggiuntive
    cum_rewards = np.cumsum(rewards)
    running_max = np.maximum.accumulate(cum_rewards)
    drawdowns = cum_rewards - running_max
    max_drawdown = np.min(drawdowns)
    
    # Calcolo delle statistiche di trading
    n_trades = sum(1 for a in actions if abs(a) > 1e-6)
    avg_position = np.mean(positions)
    avg_action = np.mean([abs(a) for a in actions])
    position_changes = [abs(positions[i] - positions[i-1]) for i in range(1, len(positions))]
    turnover = sum(position_changes)
    
    # Calcolo delle correlazioni
    if 'pred_lstm' in df_test.columns and len(df_test) == len(positions[:-1]):
        # Converte date in indici per il confronto
        position_df = pd.DataFrame({'position': positions[:-1]})  # Escludi l'ultima posizione
        
        # Trova correlazioni tra posizioni e vari indicatori
        correlations = {}
        for col in ['pred_lstm', 'pred_gru', 'pred_blstm', 'MACD', 'RSI14', 'BB_Position']:
            if col in df_test.columns:
                correlations[col] = np.corrcoef(position_df['position'], df_test[col])[0, 1]
    else:
        correlations = {}

    # Raggruppa i risultati
    results = {
        'positions': positions,
        'actions': actions,
        'rewards': rewards,
        'prices': prices,
        'dates': dates,
        'trade_costs': trade_costs,
        'position_penalties': position_penalties,
        'pnls': pnls,
        'cumulative_reward': np.sum(rewards),
        'pnl': np.sum(pnls),
        'sharpe': np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252),
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'avg_position': avg_position,
        'avg_action': avg_action,
        'turnover': turnover,
        'correlations': correlations
    }
    
    return results

# Inizializza l'ambiente con i dati di test
env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_test) - 1,
    lambd=0.05,            # Valore ridotto come nel training
    psi=0.2,               # Valore ridotto come nel training
    cost="trade_l1",
    max_pos=4,             # Valore aumentato come nel training
    squared_risk=False,
    penalty="tanh",
    alpha=3,               # Valore ridotto come nel training
    beta=3,                # Valore ridotto come nel training
    clip=True,
    scale_reward=5,        # Valore ridotto come nel training
    df=df_test,            # Usa i dati di test
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=len(df_test), # Usa tutto il dataset di test
    # Ripristina i parametri realistici di commissioni per il test
    free_trades_per_month=10,       # Valore realistico
    commission_rate=0.0025,         # Valore realistico
    min_commission=1.0              # Valore realistico
)

# Inizializza l'agente
agent = Agent()

# Analizza il miglior modello
print(f"Analisi del modello {best_model_file}...")
best_model_results = analyze_model(best_model_file, env, agent)

# Analizza il modello di confronto
print(f"Analisi del modello {comparison_model_file} per confronto...")
comparison_model_results = analyze_model(comparison_model_file, env, agent)

# Stampa statistiche dettagliate
print("\n--- STATISTICHE DETTAGLIATE ---")
print(f"Modello: {best_model_file}")
print(f"Ricompensa cumulativa: {best_model_results['cumulative_reward']:.2f}")
print(f"P&L: {best_model_results['pnl']:.2f}")
print(f"Sharpe ratio: {best_model_results['sharpe']:.2f}")
print(f"Max drawdown: {best_model_results['max_drawdown']:.2f}")
print(f"Numero di trade: {best_model_results['n_trades']}")
print(f"Posizione media: {best_model_results['avg_position']:.2f}")
print(f"Azione media: {best_model_results['avg_action']:.4f}")
print(f"Turnover totale: {best_model_results['turnover']:.2f}")
print("Correlazioni con indicatori:")
for indicator, corr in best_model_results['correlations'].items():
    print(f"  {indicator}: {corr:.4f}")

print("\nModello di confronto: {comparison_model_file}")
print(f"Ricompensa cumulativa: {comparison_model_results['cumulative_reward']:.2f}")
print(f"P&L: {comparison_model_results['pnl']:.2f}")
print(f"Sharpe ratio: {comparison_model_results['sharpe']:.2f}")
print(f"Numero di trade: {comparison_model_results['n_trades']}")
print(f"Posizione media: {comparison_model_results['avg_position']:.2f}")

# Crea visualizzazioni per confrontare i modelli
plt.figure(figsize=(14, 20))

# Plot 1: Confronto delle posizioni
plt.subplot(5, 1, 1)
plt.plot(best_model_results['dates'], best_model_results['positions'][:-1], label=f'Posizioni {best_model_file}', color='blue')
plt.plot(comparison_model_results['dates'], comparison_model_results['positions'][:-1], label=f'Posizioni {comparison_model_file}', color='red', alpha=0.7)
plt.title('Confronto delle posizioni')
plt.ylabel('Posizione')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Confronto delle azioni
plt.subplot(5, 1, 2)
plt.plot(best_model_results['dates'], best_model_results['actions'], label=f'Azioni {best_model_file}', color='blue')
plt.plot(comparison_model_results['dates'], comparison_model_results['actions'], label=f'Azioni {comparison_model_file}', color='red', alpha=0.7)
plt.title('Confronto delle azioni (trades)')
plt.ylabel('Azione')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Confronto delle ricompense cumulative
plt.subplot(5, 1, 3)
plt.plot(best_model_results['dates'], np.cumsum(best_model_results['rewards']), label=f'Ricompensa cum. {best_model_file}', color='blue')
plt.plot(comparison_model_results['dates'], np.cumsum(comparison_model_results['rewards']), label=f'Ricompensa cum. {comparison_model_file}', color='red', alpha=0.7)
plt.title('Confronto delle ricompense cumulative')
plt.ylabel('Ricompensa cumulativa')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Analisi dei costi di trading
plt.subplot(5, 1, 4)
plt.plot(best_model_results['dates'], np.cumsum(best_model_results['trade_costs']), label=f'Costi cum. {best_model_file}', color='blue')
plt.plot(comparison_model_results['dates'], np.cumsum(comparison_model_results['trade_costs']), label=f'Costi cum. {comparison_model_file}', color='red', alpha=0.7)
plt.title('Confronto dei costi di trading cumulativi')
plt.ylabel('Costi cumulativi')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Confronto del P&L
plt.subplot(5, 1, 5)
plt.plot(best_model_results['dates'], np.cumsum(best_model_results['pnls']), label=f'P&L cum. {best_model_file}', color='blue')
plt.plot(comparison_model_results['dates'], np.cumsum(comparison_model_results['pnls']), label=f'P&L cum. {comparison_model_file}', color='red', alpha=0.7)
plt.title('Confronto del P&L cumulativo')
plt.ylabel('P&L cumulativo')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/analysis/model_comparison.png')

# Visualizzazioni aggiuntive per il miglior modello
plt.figure(figsize=(14, 14))

# Plot 1: Prezzo e posizione
plt.subplot(3, 1, 1)
ax1 = plt.gca()
ax1.plot(best_model_results['dates'], best_model_results['prices'], label='Prezzo', color='green')
ax1.set_ylabel('Prezzo', color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.set_title('Prezzo vs Posizione')

ax2 = ax1.twinx()
ax2.plot(best_model_results['dates'], best_model_results['positions'][:-1], label='Posizione', color='blue')
ax2.set_ylabel('Posizione', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
plt.grid(True, alpha=0.3)

# Plot 2: Componenti della ricompensa
plt.subplot(3, 1, 2)
plt.bar(best_model_results['dates'], best_model_results['pnls'], label='P&L', color='green', alpha=0.7)
plt.bar(best_model_results['dates'], -np.array(best_model_results['trade_costs']), label='Costi di trading', color='red', alpha=0.7)
plt.bar(best_model_results['dates'], -np.array(best_model_results['position_penalties']), label='Penalità posizione', color='orange', alpha=0.7)
plt.title('Componenti della ricompensa')
plt.ylabel('Valore')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Distribuzione delle posizioni e azioni
plt.subplot(3, 1, 3)
sns.histplot(best_model_results['positions'][:-1], bins=20, kde=True, label='Posizioni', color='blue', alpha=0.5)
sns.histplot(best_model_results['actions'], bins=20, kde=True, label='Azioni', color='red', alpha=0.5)
plt.title('Distribuzione delle posizioni e azioni')
plt.xlabel('Valore')
plt.ylabel('Frequenza')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/analysis/best_model_details.png')

print(f"\nAnalisi completata. Le visualizzazioni sono state salvate in: {output_dir}/analysis/")