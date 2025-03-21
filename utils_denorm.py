import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def verifica_parametri_normalizzazione(norm_params_path, feature_name="adjClose"):
    """Stampa i valori min/max per una feature per aiutare a debuggare problemi di normalizzazione."""
    norm_params = load_normalization_params(norm_params_path)
    
    print(f"Feature disponibili: {list(norm_params['min'].keys())}")
    
    if feature_name in norm_params['min'] and feature_name in norm_params['max']:
        min_val = norm_params['min'][feature_name]
        max_val = norm_params['max'][feature_name]
        print(f"Feature: {feature_name}")
        print(f"Valore minimo: {min_val}")
        print(f"Valore massimo: {max_val}")
        
        # Testa la denormalizzazione con alcuni valori
        valori_test = [0.0, 0.25, 0.5, 0.75, 1.0]
        for val in valori_test:
            denorm_val = val * (max_val - min_val) + min_val
            print(f"Normalizzato {val} -> Denormalizzato {denorm_val}")
    else:
        print(f"Feature {feature_name} non trovata nei parametri di normalizzazione")

def load_normalization_params(json_path):
    """
    Carica i parametri di normalizzazione da un file JSON.
    
    Args:
        json_path: percorso al file JSON contenente i parametri min-max
        
    Returns:
        dict: dizionario con i parametri di normalizzazione
    """
    with open(json_path, 'r') as f:
        norm_params = json.load(f)
    return norm_params

def denormalize_value(value, feature_name, norm_params):
    """
    Denormalizza un singolo valore dato il nome della feature e i parametri di normalizzazione.
    
    Args:
        value: valore normalizzato
        feature_name: nome della feature
        norm_params: dizionario con i parametri di normalizzazione
        
    Returns:
        float: valore denormalizzato
    """
    if feature_name not in norm_params['min'] or feature_name not in norm_params['max']:
        return value  # Se non troviamo i parametri, restituiamo il valore originale
    
    min_val = norm_params['min'][feature_name]
    max_val = norm_params['max'][feature_name]
    
    # Applica la formula inversa della normalizzazione min-max
    denorm_value = value * (max_val - min_val) + min_val
    
    return denorm_value

def denormalize_series(series, feature_name, norm_params):
    """
    Denormalizza una serie di valori dato il nome della feature e i parametri di normalizzazione.
    
    Args:
        series: array o lista di valori normalizzati
        feature_name: nome della feature
        norm_params: dizionario con i parametri di normalizzazione
        
    Returns:
        numpy.ndarray: array di valori denormalizzati
    """
    if feature_name not in norm_params['min'] or feature_name not in norm_params['max']:
        return np.array(series)  # Se non troviamo i parametri, restituiamo i valori originali
    
    min_val = norm_params['min'][feature_name]
    max_val = norm_params['max'][feature_name]
    
    # Applica la formula inversa della normalizzazione min-max
    denorm_series = np.array(series) * (max_val - min_val) + min_val
    
    return denorm_series

def calculate_real_pnl(positions, price_changes, price_feature_name, norm_params):
    """
    Calcola il P&L reale dato le posizioni e le variazioni di prezzo denormalizzate.
    """
    # Denormalizza le variazioni di prezzo
    real_price_changes = denormalize_series(price_changes, price_feature_name, norm_params)
    
    # Assicura che positions e price_changes abbiano la stessa lunghezza
    min_length = min(len(positions[:-1]), len(real_price_changes))
    positions_aligned = positions[:min_length+1][:-1]  # Taglia e poi rimuovi l'ultimo
    price_changes_aligned = real_price_changes[:min_length]
    
    # Calcola il P&L giornaliero
    daily_pnl = positions_aligned * price_changes_aligned
    
    # Calcola il P&L cumulativo
    cumulative_pnl = np.cumsum(daily_pnl)
    
    return daily_pnl, cumulative_pnl

def plot_real_performance(df, positions, prices, price_feature_name="close", norm_params_path=None):
    """
    Crea un grafico che mostra la performance reale (denormalizzata) del modello.
    
    Args:
        df: DataFrame con i dati originali
        positions: array delle posizioni
        prices: array dei prezzi normalizzati
        price_feature_name: nome della feature del prezzo
        norm_params_path: percorso al file JSON con i parametri di normalizzazione
    """
    if norm_params_path:
        norm_params = load_normalization_params(norm_params_path)
        
        # Denormalizza i prezzi
        real_prices = denormalize_series(prices, price_feature_name, norm_params)
        
        # Calcola le variazioni di prezzo
        price_changes = np.diff(prices)
        
        # Calcola il P&L reale
        daily_pnl, cumulative_pnl = calculate_real_pnl(
            positions, price_changes, price_feature_name, norm_params
        )
        
        # Crea il grafico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Grafico del prezzo e della posizione
        ax1.plot(df['date'].iloc[:len(real_prices)], real_prices, color='green', label='Prezzo reale')
        ax1.set_ylabel('Prezzo ($)', color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_title('Prezzo reale vs Posizione')
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df['date'].iloc[:len(positions)-1], positions[:-1], color='blue', label='Posizione')
        ax1_twin.set_ylabel('Posizione')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        
        # Grafico del P&L cumulativo
        ax2.plot(df['date'].iloc[:len(cumulative_pnl)], cumulative_pnl, color='purple', label='P&L cumulativo ($)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('P&L cumulativo ($)')
        ax2.set_title('Performance reale (P&L cumulativo)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    else:
        # Se non abbiamo i parametri di normalizzazione, restituiamo un grafico semplice
        plt.figure(figsize=(14, 6))
        plt.plot(df['date'].iloc[:len(prices)], prices, label='Prezzo (normalizzato)')
        plt.plot(df['date'].iloc[:len(positions)-1], positions[:-1], label='Posizione')
        plt.title('Prezzo vs Posizione (valori normalizzati)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()

def estimate_trading_costs(actions, price_feature_name, norm_params, commission_rate=0.0025, min_commission=1.0):
    """
    Stima i costi di trading reali basati sulle azioni e sui prezzi denormalizzati.
    
    Args:
        actions: array delle azioni (cambiamenti di posizione)
        price_feature: array dei prezzi normalizzati corrispondenti
        price_feature_name: nome della feature del prezzo
        norm_params: dizionario con i parametri di normalizzazione
        commission_rate: tasso di commissione (default: 0.25%)
        min_commission: commissione minima (default: 1.0€)
        
    Returns:
        numpy.ndarray: array dei costi di trading giornalieri
    """
    if price_feature_name not in norm_params['min'] or price_feature_name not in norm_params['max']:
        return np.zeros_like(actions)  # Se non troviamo i parametri, restituiamo zeri
    
    # Calcola il valore medio del prezzo denormalizzato (una stima)
    price_min = norm_params['min'][price_feature_name]
    price_max = norm_params['max'][price_feature_name]
    avg_price = (price_min + price_max) / 2
    
    # Calcola i costi di trading
    costs = []
    for action in actions:
        if abs(action) > 1e-6:  # Se c'è un'azione effettiva
            order_value = abs(action) * avg_price
            cost = max(order_value * commission_rate, min_commission)
            costs.append(cost)
        else:
            costs.append(0)
    
    return np.array(costs)

def summarize_real_performance(positions, prices, actions, price_feature_name, norm_params_path,
                              commission_rate=0.0025, min_commission=1.0):
    """
    Calcola e restituisce un riepilogo della performance reale.
    
    Args:
        positions: array delle posizioni
        prices: array dei prezzi normalizzati
        actions: array delle azioni
        price_feature_name: nome della feature del prezzo
        norm_params_path: percorso al file JSON con i parametri di normalizzazione
        commission_rate: tasso di commissione
        min_commission: commissione minima
        
    Returns:
        dict: riepilogo della performance
    """
    norm_params = load_normalization_params(norm_params_path)
    
    # Denormalizza i prezzi
    real_prices = denormalize_series(prices, price_feature_name, norm_params)
    
    # Calcola le variazioni di prezzo
    price_changes = np.diff(prices)
    
    # Calcola il P&L reale
    daily_pnl, cumulative_pnl = calculate_real_pnl(
        positions, price_changes, price_feature_name, norm_params
    )
    
    # Stima i costi di trading
    trading_costs = estimate_trading_costs(
        actions, price_feature_name, norm_params, commission_rate, min_commission
    )
    
    # Calcola le metriche di performance
    total_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
    total_trading_costs = np.sum(trading_costs)
    net_profit = total_pnl - total_trading_costs
    
    # Calcola altre metriche
    n_trades = sum(1 for a in actions if abs(a) > 1e-6)
    avg_position_size = np.mean(positions)
    max_position_size = np.max(positions)
    
    # Calcola il Sharpe ratio annualizzato sul P&L giornaliero se possibile
    if len(daily_pnl) > 1:
        sharpe = np.mean(daily_pnl) / (np.std(daily_pnl) + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Calcola il drawdown massimo
    if len(cumulative_pnl) > 0:
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_max
        max_drawdown = np.min(drawdowns)
    else:
        max_drawdown = 0
    
    return {
        'total_pnl': total_pnl,
        'total_trading_costs': total_trading_costs,
        'net_profit': net_profit,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'avg_position_size': avg_position_size,
        'max_position_size': max_position_size,
        'first_price': real_prices[0] if len(real_prices) > 0 else 0,
        'last_price': real_prices[-1] if len(real_prices) > 0 else 0,
        'price_return': (real_prices[-1] / real_prices[0] - 1) * 100 if len(real_prices) > 1 else 0
    }