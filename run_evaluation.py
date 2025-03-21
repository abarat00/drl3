from evaluation import test_models, plot_bars, plot_hist, plot_function
from env import Environment
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def main():
    # Configurazione
    ticker = "ARKG"
    output_dir = f'results/{ticker}'
    weights_dir = f'{output_dir}/weights/'
    
    # Verifica che la directory dei pesi esista
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Directory dei pesi non trovata: {weights_dir}")
    
    # Definisci i parametri dell'ambiente (identici a quelli usati per il training)
    env = Environment(
        sigma=0.1,
        theta=0.1,
        T=5000,  # Dimensione generica, per l'evaluation usiamo un random_state fisso
        lambd=0.3,
        psi=0.5,
        cost="trade_l1",
        max_pos=2,
        squared_risk=False,
        penalty="tanh",
        alpha=10,
        beta=10,
        clip=True,
        scale_reward=10
    )
    
    # Valuta i modelli usando la funzione test_models
    print(f"Valutazione dei modelli in {weights_dir}...")
    scores, scores_episodes, scores_cumsum, pnls, positions = test_models(
        path_weights=weights_dir,
        env=env,
        fc1_units=128,
        fc2_units=64,
        random_state=42,  # Seed fisso per la riproducibilità
        n_episodes=10     # Numero di episodi di valutazione
    )
    
    # Crea directory per i risultati
    results_dir = f'{output_dir}/evaluation'
    os.makedirs(results_dir, exist_ok=True)
    
    # Visualizza i risultati
    print("Scores (media punteggi) per ogni modello:")
    for model_key, score in scores.items():
        print(f"  Modello {model_key}: {score:.2f}")
    
    # Genera i grafici
    plt.figure(figsize=(12, 8))
    plot_bars(scores)
    plt.title(f"Performance media dei modelli - {ticker}")
    plt.savefig(f'{results_dir}/model_scores.png')
    plt.close()
    
    # Trova il miglior modello
    best_model = max(scores.items(), key=lambda x: x[1])[0]
    print(f"Miglior modello: {best_model} con score {scores[best_model]:.2f}")
    
    # Plot distribuzione dei punteggi per il miglior modello
    plt.figure(figsize=(10, 6))
    plot_hist(model_key=best_model, scores_episodes=scores_episodes)
    plt.savefig(f'{results_dir}/best_model_distribution.png')
    plt.close()
    
    # Seleziona alcuni modelli per un confronto dettagliato
    # Prendi modelli a intervalli regolari (es. inizio, metà, fine)
    model_keys = list(sorted(k for k in scores.keys() if isinstance(k, int)))
    selected_models = []
    if len(model_keys) > 0:
        if len(model_keys) == 1:
            selected_models = model_keys
        elif len(model_keys) <= 5:
            selected_models = model_keys
        else:
            step = len(model_keys) // 5
            selected_models = [model_keys[i] for i in range(0, len(model_keys), step)][:5]
            if best_model not in selected_models and best_model in model_keys:
                selected_models[-1] = best_model
    
    # Visualizza funzione per modelli selezionati
    if selected_models:
        print(f"Generazione grafici dettagliati per i modelli: {selected_models}")
        try:
            plot_function(
                path_weights=weights_dir,
                env=env,
                models_keys=selected_models,
                low=-4,
                high=4,
                pi=0.5,
                lambd=0.3,
                psi=0.5,
                thresh=1.0,
                fc1_units=128,
                fc2_units=64,
                clip=True
            )
            plt.savefig(f'{results_dir}/model_functions.png')
            plt.close()
        except Exception as e:
            print(f"Errore nella generazione del grafico delle funzioni: {e}")
    
    # Visualizza traiettorie di trading per il miglior modello
    # Seleziona un episodio con una buona performance
    if best_model in scores_episodes:
        best_episode = max(scores_episodes[best_model].items(), key=lambda x: x[1])[0]
        print(f"Miglior episodio per il modello {best_model}: {best_episode} con score {scores_episodes[best_model][best_episode]:.2f}")
        
        # Resetta l'ambiente con il seed del miglior episodio
        env.reset(random_state=best_episode)
        
        # Crea grafico della traiettoria di trading
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.plot(env.signal, label='Segnale', color='blue', alpha=0.7)
        plt.plot(positions[best_model][best_episode], label='Posizione', color='red')
        plt.title(f"Strategia di trading - Modello {best_model}, Episodio {best_episode}")
        plt.ylabel("Valore")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(scores_cumsum[best_model][best_episode], label='Ricompensa cumulativa', color='green')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel("Timestep")
        plt.ylabel("Ricompensa cumulativa")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/best_episode_trading.png')
        plt.close()
    
    print(f"Valutazione completata. Risultati salvati in {results_dir}/")

if __name__ == "__main__":
    main()