import os
import torch
import numpy as np
from agent import Agent
from env import Environment
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Aggiungi questa importazione
from tqdm import tqdm
from datetime import datetime;

# Configurazione
ticker = "ARKG"  # Ticker da utilizzare
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'
csv_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/{ticker}/{ticker}_normalized.csv'
output_dir = f'results/{ticker}_no_commissions'

# Crea directory di output se non esiste
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/weights', exist_ok=True)
os.makedirs(f'{output_dir}/test', exist_ok=True)
os.makedirs(f'{output_dir}/analysis', exist_ok=True)

# Stampo l 'orario
print(f"Stampo l'orario")
print(datetime.now().strftime("%H:%M:%S"))
# Carica il dataset
print(f"Caricamento dati per {ticker}...")
df = pd.read_csv(csv_path)

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

# Parametri per l'ambiente
max_steps = min(1000, len(df_train) - 10)  # Limita la lunghezza massima dell'episodio
print(f"Lunghezza massima episodio: {max_steps} timestep")

# Inizializza l'ambiente (senza commissioni)
print("Inizializzazione dell'ambiente (senza commissioni)...")
# Cambia questi parametri nell'inizializzazione dell'ambiente
env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_train) - 1,
    lambd=0.01,             # Riduci questa penalità (originale: 0.05)
    psi=0.05,               # Ridotto ulteriormente (originale: 0.1)
    cost="trade_l1",
    max_pos=6.0,            # Aumenta il limite di posizione (originale: 4.0)
    squared_risk=False,
    penalty="tanh",
    alpha=1,                # Riduci (originale: 2)
    beta=1,                 # Riduci (originale: 2)
    clip=True,
    scale_reward=3,         # Riduci (originale: 5)
    df=df_train,
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=max_steps,
    free_trades_per_month=10000,
    commission_rate=0.0,
    min_commission=0.0,
    trading_frequency_penalty_factor=0.05,  # Riduci (originale: 0.1)
    position_stability_bonus_factor=0.4    # Aumenta significativamente (originale: 0.1)
)

# Parametri di training
total_episodes = 2000
save_freq = 10
learn_freq = 20

# Inizializza l'agente
print("Inizializzazione dell'agente DDPG...")
# Modifiche suggerite ai parametri dell'agente esistenti
agent = Agent(
    memory_type="prioritized",
    batch_size=512,         # Aumentato da 256 per migliorare la stabilità
    max_step=max_steps,
    theta=0.03,             # Ridotto ulteriormente da 0.05 per un'esplorazione più mirata
    sigma=0.35              # Leggermente ridotto da 0.4 ma ancora alto per garantire esplorazione
)

# Modifiche suggerite ai train_params esistenti
train_params = {
    'tau_actor': 0.005,     # Ridotto da 0.01 per aggiornamenti più graduali
    'tau_critic': 0.02,     # Ridotto da 0.05 per aggiornamenti più graduali
    'lr_actor': 5e-6,       # Ridotto da 1e-5 per convergenza più stabile
    'lr_critic': 1e-4,      # Ridotto da 2e-4 per convergenza più stabile
    'weight_decay_actor': 1e-6,  # Mantenuto uguale
    'weight_decay_critic': 2e-5, # Mantenuto uguale
    'total_steps': 3000,    # Aumentato da 2000 per un pretraining più lungo
    'weights': f'{output_dir}/weights/',
    'freq': save_freq,
    'fc1_units_actor': 128,
    'fc2_units_actor': 64,
    'fc1_units_critic': 256,
    'fc2_units_critic': 128,
    'learn_freq': 10,       # Ridotto da 20 per aggiornare i pesi più frequentemente
    'decay_rate': 5e-7,     # Ridotto da 1e-6 per mantenere l'esplorazione più a lungo
    'explore_stop': 0.05,   # Ridotto da 0.1 per consentire un'esplorazione minima più ridotta
    'tensordir': f'{output_dir}/runs/',
    'progress': "tqdm", 
}

# Avvia il training
print(f"Avvio del training per {ticker} - {total_episodes} episodi (senza commissioni)...")
agent.train(
    env=env,
    total_episodes=total_episodes,
    **train_params
)

print(f"Training completato per {ticker}!")
print(f"I modelli addestrati sono stati salvati in: {output_dir}/weights/")
print(f"I log per TensorBoard sono stati salvati in: {output_dir}/runs/")

# Valutazione sul dataset di test
print("\nAvvio della valutazione sul dataset di test...")

from utils_denorm import verifica_parametri_normalizzazione
verifica_parametri_normalizzazione(norm_params_path)

# Funzione per valutare un modello
# Modifica alla funzione evaluate_model per calcolare il P&L e altre metriche
def evaluate_model(model_file, env, agent, df_test, norm_params_path):
    """Valuta le performance di un modello sul dataset di test con prezzi denormalizzati."""
    model_path = os.path.join(f'{output_dir}/weights/', model_file)
    
    # Importa le funzioni di denormalizzazione
    from utils_denorm import load_normalization_params, denormalize_series
    
    # Carica i parametri di normalizzazione
    norm_params = load_normalization_params(norm_params_path)
    
    # Carica il modello
    from models import Actor
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
    norm_prices = []
    dates = []
    
    # Esegui un singolo episodio attraverso tutti i dati di test
    while not done:
        action = agent.act(state, noise=False)  # Nessun rumore durante il test
        actions.append(action)
        
        # Registra lo stato prima dell'azione
        current_price = env.p
        norm_prices.append(current_price)
        
        if 'date' in df_test.columns and env.current_index < len(df_test):
            dates.append(df_test['date'].iloc[env.current_index])
        
        # Esegui l'azione e ottieni la ricompensa
        reward = env.step(action)
        rewards.append(reward)
        
        # Aggiorna lo stato
        state = env.get_state()
        positions.append(env.pi)
        done = env.done
    
    # Denormalizza i prezzi (assumendo che il prezzo sia nella feature "Log_Close" o simile)
    price_feature = "adjClose"  # Modifica in base alla feature che contiene il prezzo
    if price_feature not in norm_params['min'] or price_feature not in norm_params['max']:
        # Cerca altre feature di prezzo comuni
        for feature in ["close", "adjClose", "price"]:
            if feature in norm_params['min'] and feature in norm_params['max']:
                price_feature = feature
                break
    
    # Denormalizza i prezzi usando i parametri corretti
    real_prices = denormalize_series(norm_prices, price_feature, norm_params)
    
    # Calcola i cambiamenti di prezzo reali
    real_price_changes = np.diff(real_prices)
    
    # Calcola il P&L usando i prezzi reali e le posizioni
    pnl_values = []
    for i in range(len(real_price_changes)):
        pnl_values.append(positions[i] * real_price_changes[i])
    
    total_pnl = np.sum(pnl_values)
    cum_pnl = np.cumsum(pnl_values)
    
    # Calcola metriche di performance
    cumulative_reward = np.sum(rewards)
    
    # Calcola altre metriche
    if len(rewards) > 1:
        sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)  # Annualizzato
        cum_rewards = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cum_rewards)
        drawdowns = cum_rewards - running_max
        max_drawdown = np.min(drawdowns)
        
        # Calcola il Sortino ratio (considera solo i rendimenti negativi per la volatilità)
        negative_returns = [r for r in pnl_values if r < 0]
        sortino = np.mean(pnl_values) / (np.std(negative_returns) + 1e-8) * np.sqrt(252)
        
        # Calcola il Maximum Drawdown in termini percentuali per il P&L
        if len(cum_pnl) > 0:
            running_max_pnl = np.maximum.accumulate(cum_pnl)
            # Evita la divisione per zero
            valid_indices = running_max_pnl != 0
            if np.any(valid_indices):
                pnl_drawdowns = (cum_pnl - running_max_pnl) / np.maximum(running_max_pnl, 1e-8)
                max_pnl_drawdown_pct = np.min(pnl_drawdowns[valid_indices]) * 100
            else:
                max_pnl_drawdown_pct = 0
        else:
            max_pnl_drawdown_pct = 0
    else:
        sharpe = 0
        max_drawdown = 0
        sortino = 0
        max_pnl_drawdown_pct = 0
    
    # Calcola la metrica di confronto con buy-and-hold su prezzi reali
    if len(real_prices) >= 2:
        # Rendimento buy-and-hold: comprare all'inizio e vendere alla fine
        buy_and_hold_return = (real_prices[-1] - real_prices[0]) / real_prices[0]
        
        # Calcola l'excess return rispetto a buy-and-hold
        strategy_return = total_pnl / real_prices[0]  # Rendimento della strategia (normalizzato)
        excess_return = strategy_return - buy_and_hold_return
        
        # Information ratio (eccesso di rendimento diviso per tracking error)
        # Calcola i rendimenti giornalieri della strategia e del buy-and-hold
        strategy_daily_returns = np.array(pnl_values) / real_prices[0]
        bh_daily_price_changes = np.diff(real_prices)
        bh_daily_returns = bh_daily_price_changes / real_prices[:-1]
        
        # Assicurati che abbiano la stessa lunghezza
        min_len = min(len(strategy_daily_returns), len(bh_daily_returns))
        if min_len > 1:
            # Calcola il tracking error (std dev della differenza tra i rendimenti)
            return_diff = strategy_daily_returns[:min_len] - bh_daily_returns[:min_len]
            tracking_error = np.std(return_diff) * np.sqrt(252)
            information_ratio = excess_return / (tracking_error + 1e-8)
        else:
            information_ratio = 0
    else:
        buy_and_hold_return = 0
        excess_return = 0
        information_ratio = 0
    
    # Calcola la percentuale di giorni con trade profittevoli
    profitable_days = sum(1 for p in pnl_values if p > 0)
    if len(pnl_values) > 0:
        profitable_days_pct = profitable_days / len(pnl_values) * 100
    else:
        profitable_days_pct = 0
    
    # Calcola il Calmar ratio (rendimento medio annualizzato diviso per max drawdown)
    daily_return = np.mean(pnl_values) if pnl_values else 0
    annualized_return = daily_return * 252
    calmar_ratio = annualized_return / (abs(max_pnl_drawdown_pct) + 1e-8)
    
    # Analisi del comportamento di trading
    n_trades = sum(1 for a in actions if abs(a) > 1e-6)
    avg_position = np.mean(positions)
    turnover = sum(abs(positions[i] - positions[i-1]) for i in range(1, len(positions)))
    
    # Win/Loss ratio
    wins = sum(1 for p in pnl_values if p > 0)
    losses = sum(1 for p in pnl_values if p < 0)
    win_loss_ratio = wins / (losses + 1e-8)  # Evita divisione per zero

    # Calcolo del rendimento percentuale totale della strategia
    initial_investment = real_prices[0]  # Utilizziamo il prezzo iniziale come capitale iniziale
    pct_return = (total_pnl / initial_investment) * 100  # Rendimento percentuale
    
    # Ritorna anche i prezzi reali e altre informazioni utili
    return {
        'positions': positions,
        'actions': actions,
        'rewards': rewards,
        'norm_prices': norm_prices,
        'real_prices': real_prices,
        'dates': dates,
        'pnl_values': pnl_values,
        'cumulative_reward': cumulative_reward,
        'total_pnl': total_pnl,
        'pct_return': pct_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'max_pnl_drawdown_pct': max_pnl_drawdown_pct,
        'buy_and_hold_return': buy_and_hold_return * 100,  # Convertito in percentuale
        'excess_return': excess_return * 100,  # Convertito in percentuale
        'information_ratio': information_ratio,
        'profitable_days_pct': profitable_days_pct,
        'calmar_ratio': calmar_ratio,
        'n_trades': n_trades,
        'avg_position': avg_position,
        'turnover': turnover,
        'win_loss_ratio': win_loss_ratio
    }

# Inizializza l'ambiente di test
test_env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_test) - 1,
    lambd=0.05,
    psi=0.1,
    cost="trade_l1",
    max_pos=4.0,
    squared_risk=False,
    penalty="tanh",
    alpha=2,
    beta=2,
    clip=True,
    scale_reward=5,
    df=df_test,
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=len(df_test),
    # Parametri senza commissioni anche per il test
    free_trades_per_month=10000,
    commission_rate=0.0,
    min_commission=0.0,
    trading_frequency_penalty_factor=0.1,
    position_stability_bonus_factor=0.1
)

# Sezione di codice che chiama la funzione evaluate_model
# Nella sezione di valutazione del codice

# Ottieni la lista dei modelli salvati
weights_dir = f'{output_dir}/weights'
model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
model_files.sort(key=lambda x: int(x[4:-4]) if x[4:-4].isdigit() else 0)

if not model_files:
    print("Nessun modello trovato per la valutazione.")
else:
    # Scegli modelli da valutare (primo, medio, ultimo)
    if len(model_files) > 3:
        selected_models = [
            model_files[0],  # primo modello
            model_files[len(model_files) // 2],  # modello a metà
            model_files[-1]  # ultimo modello
        ]
    else:
        selected_models = model_files
    
    # Valuta i modelli selezionati
    evaluation_results = []
    
    for model_file in selected_models:
        print(f"Valutazione del modello {model_file}...")
        # Passa i parametri aggiuntivi: df_test e norm_params_path
        results = evaluate_model(model_file, test_env, agent, df_test, norm_params_path)
        results['model'] = model_file
        evaluation_results.append(results)
        
        # Stampa i risultati con più dettagli
        print(f"  Ricompensa cumulativa: {results['cumulative_reward']:.2f}")
        print(f"  P&L totale (denormalizzato): ${results['total_pnl']:.2f}")
        print(f"  Rendimento percentuale: {results['pct_return']:.2f}%")
        print(f"  Sharpe ratio: {results['sharpe']:.2f}")
        print(f"  Sortino ratio: {results['sortino']:.2f}")
        print(f"  Max drawdown (ricompensa): {results['max_drawdown']:.2f}")
        print(f"  Max drawdown P&L: {results['max_pnl_drawdown_pct']:.2f}%")
        print(f"  Return buy-and-hold: {results['buy_and_hold_return']:.2f}%")
        print(f"  Excess return: {results['excess_return']:.2f}%")
        print(f"  Information ratio: {results['information_ratio']:.2f}")
        print(f"  Giorni profittevoli: {results['profitable_days_pct']:.2f}%")
        print(f"  Win/Loss ratio: {results['win_loss_ratio']:.2f}")
        print(f"  Calmar ratio: {results['calmar_ratio']:.2f}")
        print(f"  Numero di trade: {results['n_trades']}")
        print(f"  Posizione media: {results['avg_position']:.2f}")
        print(f"  Turnover totale: {results['turnover']:.2f}")
        
        # Aggiungi informazione sui prezzi denormalizzati per verificare
        print(f"  Prezzo iniziale: ${results['real_prices'][0]:.2f}")
        print(f"  Prezzo finale: ${results['real_prices'][-1]:.2f}")
        print("")

# Salva i risultati
# Salva i risultati
if evaluation_results:
    # Converti i risultati in DataFrame, escludendo le liste e gli array
    eval_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list) and not isinstance(v, np.ndarray)} for r in evaluation_results])
    eval_df.to_csv(f"{output_dir}/test/evaluation_results.csv", index=False)
    print(f"Risultati della valutazione salvati in: {output_dir}/test/evaluation_results.csv")
    
    # Trova il miglior modello basato sul P&L totale anziché sulla ricompensa cumulativa
    best_model_idx = np.argmax([r['total_pnl'] for r in evaluation_results])
    best_model = evaluation_results[best_model_idx]
    
    print(f"\nMiglior modello basato sul P&L: {best_model['model']}")
    print(f"  P&L totale: ${best_model['total_pnl']:.2f}")
    print(f"  Rendimento percentuale: {best_model['pct_return']:.2f}%")
    print(f"  Ricompensa cumulativa: {best_model['cumulative_reward']:.2f}")
    print(f"  Sharpe ratio: {best_model['sharpe']:.2f}")
    print(f"  Return buy-and-hold: {best_model['buy_and_hold_return']:.2f}%")
    print(f"  Excess return vs. buy-and-hold: {best_model['excess_return']:.2f}%")
    
    # Crea un grafico del miglior modello con informazioni aggiuntive
    plt.figure(figsize=(14, 15))
    
    # Plot delle posizioni
    plt.subplot(5, 1, 1)
    plt.plot(best_model['positions'][:-1], label='Posizione', color='blue')
    plt.title(f'Performance del modello {best_model["model"]}')
    plt.ylabel('Posizione')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot delle azioni
    plt.subplot(5, 1, 2)
    plt.plot(best_model['actions'], label='Azioni', color='red')
    plt.title('Azioni (trades)')
    plt.ylabel('Azioni')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot dei prezzi reali
    plt.subplot(5, 1, 3)
    plt.plot(best_model['real_prices'], label='Prezzo reale ($)', color='green')
    plt.title('Prezzi reali (denormalizzati)')
    plt.ylabel('Prezzo ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot della ricompensa cumulativa
    plt.subplot(5, 1, 4)
    plt.plot(np.cumsum(best_model['rewards']), label='Ricompensa cumulativa', color='purple')
    plt.title('Ricompensa cumulativa')
    plt.ylabel('Ricompensa cumulativa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Modifica nella parte del grafico che genera il P&L cumulativo
    plt.subplot(5, 1, 5)

    # Prima definisci cum_pnl e buy_hold_equity
    cum_pnl = np.cumsum(best_model['pnl_values'])
    initial_price = best_model['real_prices'][0]
    buy_hold_equity = [(p - initial_price) for p in best_model['real_prices'][1:len(cum_pnl)+1]]

    # Usa le date per il plot
    if len(best_model['dates']) >= len(cum_pnl):
        dates = best_model['dates'][:len(cum_pnl)]
        plt.plot(dates, cum_pnl, label=f'P&L Cumulativo (${best_model["total_pnl"]:.2f})', color='purple')
        plt.plot(dates, buy_hold_equity, label=f'Buy & Hold ({best_model["buy_and_hold_return"]:.2f}%)', color='orange', linestyle='--')
        
        # Formatta l'asse X per mostrare SOLO mese e anno (non il giorno)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Solo mese e anno
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Mostra ogni 3 mesi
    else:
        # Fallback se non ci sono abbastanza date
        plt.plot(cum_pnl, label=f'P&L Cumulativo (${best_model["total_pnl"]:.2f})', color='purple')
        plt.plot(buy_hold_equity, label=f'Buy & Hold ({best_model["buy_and_hold_return"]:.2f}%)', color='orange', linestyle='--')

    plt.xticks(rotation=45)  # Ruota le etichette dell'asse x
    plt.title('P&L Cumulativo vs Buy & Hold (denormalizzato)')
    plt.xlabel('Data')
    plt.ylabel('P&L ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Aggiungi più spazio in basso per le etichette dell'asse X
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Dai più spazio alle etichette
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/test/best_model_performance.png")
    print(f"Grafico delle performance salvato in: {output_dir}/test/best_model_performance.png")
    
    # Aggiungiamo un grafico di confronto tra tutti i modelli
    plt.figure(figsize=(14, 10))
    
    # Confronto del P&L cumulativo tra i modelli
    plt.subplot(2, 2, 1)
    for i, result in enumerate(evaluation_results):
        cum_pnl = np.cumsum(result['pnl_values'])
        plt.plot(cum_pnl, label=f"{result['model']} (${result['total_pnl']:.2f})")
    
    # Buy & Hold come riferimento
    best_real_prices = evaluation_results[best_model_idx]['real_prices']
    initial_price = best_real_prices[0]
    buy_hold_equity = [(p - initial_price) for p in best_real_prices[1:len(cum_pnl)+1]]
    plt.plot(buy_hold_equity, label=f'Buy & Hold ({best_model["buy_and_hold_return"]:.2f}%)', color='black', linestyle='--')
    
    plt.title('Confronto P&L Cumulativo tra Modelli (denormalizzato)')
    plt.xlabel('Timestep')
    plt.ylabel('P&L ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confronto delle metriche chiave
    plt.subplot(2, 2, 2)
    model_names = [r['model'] for r in evaluation_results]
    total_pnls = [r['total_pnl'] for r in evaluation_results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x, total_pnls, width, label='P&L Totale ($)')
    
    # Aggiungi la linea del rendimento buy-and-hold
    bh_return = best_model['buy_and_hold_return'] * initial_price / 100
    plt.axhline(y=bh_return, color='r', linestyle='--', label=f'Buy & Hold: ${bh_return:.2f}')
    
    plt.xlabel('Modello')
    plt.ylabel('P&L ($)')
    plt.title('P&L Totale per Modello (denormalizzato)')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics Comparison: Sharpe, Sortino, Information Ratio
    plt.subplot(2, 2, 3)
    metrics = ['sharpe', 'sortino', 'information_ratio', 'calmar_ratio']
    metrics_labels = ['Sharpe', 'Sortino', 'Info Ratio', 'Calmar']
    
    x = np.arange(len(metrics_labels))
    width = 0.2
    multiplier = 0
    
    for i, result in enumerate(evaluation_results):
        offset = width * multiplier
        values = [result[m] for m in metrics]
        plt.bar(x + offset, values, width, label=result['model'])
        multiplier += 1
    
    plt.xlabel('Metrica')
    plt.ylabel('Valore')
    plt.title('Confronto Metriche di Performance')
    plt.xticks(x + width, metrics_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Win rate and profitable days
    plt.subplot(2, 2, 4)
    metrics = ['profitable_days_pct', 'win_loss_ratio']
    metrics_labels = ['% Giorni Profittevoli', 'Win/Loss Ratio']
    
    x = np.arange(len(metrics_labels))
    width = 0.2
    multiplier = 0
    
    for i, result in enumerate(evaluation_results):
        offset = width * multiplier
        # Limita win_loss_ratio a 10 per visualizzazione
        win_loss = min(result['win_loss_ratio'], 10)
        values = [result['profitable_days_pct'], win_loss]
        plt.bar(x + offset, values, width, label=result['model'])
        multiplier += 1
    
    plt.xlabel('Metrica')
    plt.ylabel('Valore')
    plt.title('Metriche di Successo Trading')
    plt.xticks(x + width, metrics_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/test/models_comparison_metrics.png")
    print(f"Grafico di confronto delle metriche salvato in: {output_dir}/test/models_comparison_metrics.png")
    
    
    # Aggiungiamo un grafico di confronto tra tutti i modelli
    plt.figure(figsize=(14, 10))
    
    # Confronto del P&L cumulativo tra i modelli
    plt.subplot(2, 2, 1)
    for i, result in enumerate(evaluation_results):
        cum_pnl = np.cumsum(result['pnl_values'])
        plt.plot(cum_pnl, label=f"{result['model']} (${result['total_pnl']:.2f})")
    
    # Buy & Hold come riferimento
    best_real_prices = evaluation_results[best_model_idx]['real_prices']
    initial_price = best_real_prices[0]
    buy_hold_values = [(p - initial_price) for p in best_real_prices[1:len(cum_pnl)+1]]
    plt.plot(buy_hold_values, label=f'Buy & Hold ({best_model["buy_and_hold_return"]:.2f}%)', color='black', linestyle='--')
    
    plt.title('Confronto P&L Cumulativo tra Modelli')
    plt.xlabel('Timestep')
    plt.ylabel('P&L ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confronto delle metriche chiave
    plt.subplot(2, 2, 2)
    model_names = [r['model'] for r in evaluation_results]
    total_pnls = [r['total_pnl'] for r in evaluation_results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x, total_pnls, width, label='P&L Totale ($)')
    
    # Aggiungi la linea del rendimento buy-and-hold
    bh_value = (best_model['buy_and_hold_return'] / 100) * initial_price
    plt.axhline(y=bh_value, color='r', linestyle='--', label=f'Buy & Hold: ${bh_value:.2f}')
    
    plt.xlabel('Modello')
    plt.ylabel('P&L ($)')
    plt.title('P&L Totale per Modello')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confronto dei rendimenti percentuali
    plt.subplot(2, 2, 3)
    returns = [r['pct_return'] for r in evaluation_results]
    bh_return = best_model['buy_and_hold_return']
    
    plt.bar(x, returns, width, label='Rendimento Strategia (%)')
    plt.axhline(y=bh_return, color='r', linestyle='--', label=f'Buy & Hold: {bh_return:.2f}%')
    
    plt.xlabel('Modello')
    plt.ylabel('Rendimento (%)')
    plt.title('Rendimento Percentuale per Modello')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confronto dei rapporti di performance
    plt.subplot(2, 2, 4)
    metrics = ['sharpe', 'sortino', 'information_ratio']
    metrics_labels = ['Sharpe', 'Sortino', 'Info Ratio']
    
    x = np.arange(len(metrics_labels))
    width = 0.2
    multiplier = 0
    
    for i, result in enumerate(evaluation_results):
        offset = width * multiplier
        values = [result[m] for m in metrics]
        # Limita i valori per una migliore visualizzazione
        values = [max(min(v, 10), -10) for v in values]
        plt.bar(x + offset, values, width, label=result['model'])
        multiplier += 1
    
    plt.xlabel('Metrica')
    plt.ylabel('Valore')
    plt.title('Confronto Metriche di Performance')
    plt.xticks(x + width, metrics_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/test/models_comparison_metrics.png")
    print(f"Grafico di confronto delle metriche salvato in: {output_dir}/test/models_comparison_metrics.png")