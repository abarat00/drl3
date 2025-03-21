import numpy as np
import json
from utils import build_ou_process

class Environment:
    """
    L'ambiente rappresenta il problema di ottimizzazione del portafoglio.
    Lo stato è costituito dalle feature già normalizzate (disponibili in self.raw_state)
    riordinate secondo l'ordine definito in self.norm_columns, concatenate con la
    posizione corrente (pi). Il segnale viene generato tramite un processo di Ornstein-Uhlenbeck.
    """
    def __init__(
        self,
        sigma=0.5,
        theta=1.0,
        T=1000,
        random_state=None,
        lambd=0.5,
        psi=0.5,
        cost="trade_0",
        max_pos=10,
        squared_risk=True,
        penalty="none",
        alpha=10,
        beta=10,
        clip=True,
        noise=False,
        noise_std=10,
        noise_seed=None,
        scale_reward=10,
        df=None,
        max_step=100,
        norm_params_path=None,   # Percorso al file JSON con i parametri Min-Max
        norm_columns=None,       # Lista delle colonne (feature) da utilizzare, in ordine
        free_trades_per_month=10,  # Numero di operazioni gratuite al mese
        commission_rate=0.0025,    # Commissione percentuale (0.25%)
        min_commission=1.0,        # Commissione minima
        trading_frequency_penalty_factor=0.0,  # Fattore di penalità per trading frequente
        position_stability_bonus_factor=0.0,    # Fattore di bonus per stabilità posizione
        initial_capital=10000
    ):
        self.raw_state_history = []
        self.price_history = []
        self.sigma = sigma
        self.theta = theta
        self.T = T
        self.lambd = lambd
        self.psi = psi
        self.cost = cost
        self.max_pos = max_pos
        self.squared_risk = squared_risk
        self.random_state = random_state
        self.signal = build_ou_process(T, sigma, theta, random_state)
        self.it = 0
        self.pi = 0
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)  # Stato "base" (vecchio formato)
        self.done = False
        self.action_size = 1
        self.penalty = penalty
        self.alpha = alpha
        self.beta = beta
        self.clip = clip
        self.scale_reward = scale_reward
        self.noise = noise
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.df = df
        self.current_index = 0
        # Parametri per le commissioni di trading
        self.free_trades_per_month = free_trades_per_month
        self.trade_count = 0  # Contatore delle operazioni effettuate
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.month_start_index = 0  # Indice di inizio del mese corrente
        self.current_month = None  # Per tenere traccia del mese corrente
        
        # Nuovi parametri per controllare il comportamento di trading
        self.trading_frequency_penalty_factor = trading_frequency_penalty_factor
        self.position_stability_bonus_factor = position_stability_bonus_factor

        # Nuovi attributi per il tracking del portafoglio
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Inizializzazione delle liste per tracciare lo storico
        self.position_history = []
        self.action_history = []
        
        if df is not None:
            self.T = len(df) - 1 # La lunghezza massima sarà il dataframe
        if noise:
            if noise_seed is None:
                self.noise_array = np.random.normal(0, noise_std, T)
            else:
                rng = np.random.RandomState(noise_seed)
                self.noise_array = rng.normal(0, noise_std, T)

        # Carica i parametri di normalizzazione, se fornito
        if norm_params_path is not None:
            with open(norm_params_path, 'r') as f:
                self.norm_params = json.load(f)
        else:
            self.norm_params = None

        # Definisci l'ordine esatto delle feature da utilizzare.
        if norm_columns is not None:
            self.norm_columns = norm_columns
        else:
            # Esempio: le 64 feature da utilizzare
            self.norm_columns = [
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
        # La dimensione dello stato è il numero di feature + 1 (per la posizione)
        self.state_size = len(self.norm_columns) + 1

        # Inizializza raw_state come dizionario con chiavi definite da norm_columns.
        # Questo verrà aggiornato ad ogni step con i dati correnti (già normalizzati)
        self.raw_state = {col: 0.0 for col in self.norm_columns}

    def update_raw_state(self, df, current_index):
        """
        Aggiorna self.raw_state leggendo la riga corrente dal DataFrame df.
        Il DataFrame df deve contenere le colonne definite in self.norm_columns.
        """
        row = df.iloc[current_index].to_dict()
        missing = [col for col in self.norm_columns if col not in row]
        if missing:
            raise ValueError(f"Mancano le seguenti colonne nel DataFrame: {missing}")
        
        # Salva lo stato corrente nella cronologia prima di aggiornarlo
        if hasattr(self, 'raw_state'):
            self.raw_state_history.append(self.raw_state.copy())
        
        # Aggiorna lo stato corrente
        self.raw_state = row
        
        # Aggiungi il prezzo alla cronologia
        if "adjClose" in self.raw_state:
            self.price_history.append(self.raw_state["adjClose"])
        elif "Log_Close" in self.raw_state:
            self.price_history.append(self.raw_state["Log_Close"])
        elif "close" in self.raw_state:
            self.price_history.append(self.raw_state["close"])

    def reset(self, random_state=None, noise_seed=None, start_index=None):
        """
        Resetta l'ambiente per iniziare un nuovo episodio.
        
        Parametri:
        - random_state: seme per la generazione del processo OU (se utilizzato)
        - noise_seed: seme per il rumore (se utilizzato)
        - start_index: indice iniziale nel DataFrame (se si usano dati reali)
        
        Returns:
        - Vettore di stato iniziale
        """
        # Reset del processo OU per compatibilità con il vecchio codice
        self.signal = build_ou_process(self.T, self.sigma, self.theta, random_state)
        self.it = 0
        self.pi = 0
        self.cash = self.initial_capital  # Reset del cash
         # Reset delle liste di cronologia
        self.raw_state_history = []
        self.price_history = []
        
        # Gestione dell'indice nei dati reali
        if self.df is not None:
            if start_index is not None:
                # Se specificato un indice di partenza, usalo (con limite di sicurezza)
                self.current_index = min(start_index, len(self.df) - self.max_step - 1)
            else:
                # Altrimenti possiamo partire da un punto casuale o dall'inizio
                if random_state is not None:
                    rng = np.random.RandomState(random_state)
                    self.current_index = rng.randint(0, max(1, len(self.df) - self.max_step - 1))
                else:
                    self.current_index = 0
            
            # Aggiorna lo stato raw con i dati reali
            self.update_raw_state(self.df, self.current_index)
            
            # Usa il prezzo di chiusura come segnale
            if "Log_Close" in self.raw_state:
                self.p = self.raw_state["Log_Close"]
            elif "close" in self.raw_state:
                self.p = self.raw_state["close"]
            else:
                # Fallback su un altro valore se necessario
                self.p = 0.0
        else:
            # Usa il processo OU come segnale simulato
            self.p = self.signal[self.it + 1]
        
        # Reset delle variabili di tracking
        self.position_history = []
        self.action_history = []
        
        # Resetta la variabile di fine episodio
        self.done = False
        
        # Gestione del rumore (per compatibilità con il vecchio codice)
        if self.noise:
            if noise_seed is None:
                self.noise_array = np.random.normal(0, self.noise_std, self.T)
            else:
                rng = np.random.RandomState(noise_seed)
                self.noise_array = rng.normal(0, self.noise_std, self.T)
        
        # Aggiorna lo stato base per compatibilità
        self.state = (self.p, self.pi)
    
        return self.get_state()
    # In env.py, aggiungi questa funzione alla classe Environment

    def calculate_risk(self, position, volatility_window=20):
        """Calcola una metrica di rischio basata sulla volatilità recente"""
        if len(self.price_history) < 2:
            return 0
        
        # Calcola la volatilità recente usando le variazioni di prezzo
        window_size = min(volatility_window, len(self.price_history))
        if window_size < 2:
            return 0
            
        # Prendi gli ultimi N prezzi
        recent_prices = self.price_history[-window_size:]
        
        # Calcola i rendimenti
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        if not price_changes:
            return 0
        
        # Calcola la volatilità come deviazione standard
        volatility = np.std(price_changes) * np.sqrt(252)  # Annualizzata
        
        # Rischio è proporzionale a posizione * volatilità
        risk = abs(position) * volatility * 5  # Fattore di scala
        
        return risk


    def get_state(self):
        """
        Restituisce lo stato corrente come un vettore ottenuto concatenando le feature
        (estratte da self.raw_state in base a self.norm_columns) e la posizione corrente (pi).

        Ritorna:
            Un array NumPy di dimensione (state_size,).
        """
        # Assicura che tutte le chiavi attese siano presenti in raw_state
        missing_cols = [col for col in self.norm_columns if col not in self.raw_state]
        if missing_cols:
            raise ValueError(f"Mancano le seguenti colonne in raw_state: {missing_cols}")

        ordered_features = [self.raw_state[col] for col in self.norm_columns]
        ordered_features.append(self.pi)
        return np.array(ordered_features)

    def step(self, action):
        """
        Applica un'azione nell'ambiente, avanza al timestep successivo e calcola la ricompensa.

        Parametri:
        - action: azione da eseguire (cambio di posizione)

        Returns:
        - Ricompensa ottenuta dall'azione
        """
        assert not self.done, "L'episodio è terminato. Resetta l'ambiente prima di procedere."
        
        # Salva posizione e prezzo correnti
        pi_prev = self.pi
        current_price = self.p
        
        # Calcola la nuova posizione dopo l'azione
        pi_next_unclipped = pi_prev + action
        if self.clip:
            pi_next = np.clip(pi_next_unclipped, -self.max_pos, self.max_pos)
        else:
            pi_next = pi_next_unclipped
        
        # Calcola la penalità per le posizioni fuori limite
        if self.penalty == "none":
            pen = 0
        elif self.penalty == "constant":
            pen = self.alpha * max(
                0,
                (self.max_pos - pi_next) / abs(self.max_pos - pi_next) if self.max_pos != pi_next else 0,
                (-self.max_pos - pi_next) / abs(-self.max_pos - pi_next) if -self.max_pos != pi_next else 0,
            )
        elif self.penalty == "tanh":
            pen = self.beta * (np.tanh(self.alpha * (abs(pi_next_unclipped) - 5 * self.max_pos / 4)) + 1)
        elif self.penalty == "exp":
            pen = self.beta * np.exp(self.alpha * (abs(pi_next) - self.max_pos))
        
        # Traccia lo storico delle posizioni per calcolare la stabilità
        if not hasattr(self, 'position_history'):
            self.position_history = []
        self.position_history.append(pi_prev)
        
        # Traccia lo storico delle azioni per calcolare la frequenza di trading
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(action)
        
        # Calcola la penalità per il trading frequente
        # Più trading recente = penalità maggiore
        trading_frequency_penalty = 0
        if hasattr(self, 'trading_frequency_penalty_factor') and self.trading_frequency_penalty_factor > 0:
            # Considera solo le ultime n azioni (default: 5)
            history_window = 5
            recent_actions = self.action_history[-history_window:] if len(self.action_history) >= history_window else self.action_history
            # Conta quante azioni significative ci sono state di recente
            significant_actions = sum(1 for a in recent_actions if abs(a) > 1e-6)
            # Calcola la penalità in base alla frequenza di trading
            trading_frequency_penalty = self.trading_frequency_penalty_factor * (significant_actions / len(recent_actions))
        # Incorpora anche altri indicatori come RSI e inclinazione delle medie mobili
        if "RSI14" in self.raw_state:
            rsi = self.raw_state["RSI14"]
            trend_reward = 0
            # Premia posizioni short quando RSI è alto (>70)
            if pi_next < 0 and rsi > 0.7:
                trend_reward += abs(pi_next) * (rsi - 0.5) * 0.15
            # Premia posizioni long quando RSI è basso (<30)
            elif pi_next > 0 and rsi < 0.3:
                trend_reward += abs(pi_next) * (0.5 - rsi) * 0.15
        # Calcola il bonus per la stabilità della posizione
        stability_bonus = 0
        if hasattr(self, 'position_stability_bonus_factor') and self.position_stability_bonus_factor > 0:
            # Se l'azione è piccola, diamo un bonus (premiamo il non-trading)
            if abs(action) < 0.05:  # Soglia per considerare un'azione come "piccola"
                stability_bonus = self.position_stability_bonus_factor
        
        # Aggiorna la posizione
        self.pi = pi_next
        
        # Gestione dei dati temporali e calcolo della ricompensa
        if self.df is not None:
            # Avanza al prossimo timestep nel DataFrame
            self.current_index += 1
            
            # Controlla se siamo arrivati alla fine del dataset
            if self.current_index >= len(self.df) - 1:
                self.done = True
                # Ricompensa finale (potrebbe essere personalizzata)
                return 0
            
            # Aggiorna lo stato con i nuovi dati
            self.update_raw_state(self.df, self.current_index)
            
            # Gestione del cambio mese e conteggio delle operazioni
            if 'date' in self.df.columns:
                current_date_str = self.df['date'].iloc[self.current_index]
                
                # Converti da stringa a datetime se necessario
                if isinstance(current_date_str, str):
                    from datetime import datetime
                    current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
                else:
                    # Assume che sia già un datetime
                    current_date = current_date_str
                
                current_month = (current_date.year, current_date.month)
                
                # Reset contatore operazioni se cambia il mese
                if current_month != self.current_month:
                    self.current_month = current_month
                    self.trade_count = 0
            
            # Ottieni il nuovo prezzo e calcola il rendimento
            if "adjClose" in self.raw_state:
                next_price = self.raw_state["adjClose"]
                price_return = next_price - current_price  # Rendimento logaritmico
            elif "close" in self.raw_state:
                next_price = self.raw_state["close"]
                if current_price != 0:
                    price_return = ((next_price / current_price) - 1) * 100  # Rendimento percentuale
                else:
                    price_return = 0
            else:
                # Usa una colonna alternativa se necessario
                next_price = current_price + self.raw_state.get("change", 0.0)
                price_return = self.raw_state.get("change", 0.0)
            
            # Aggiorna il prezzo corrente
            self.p = next_price
            
            # 1. Profitto/Perdita dalla posizione
            pnl = pi_prev * price_return
            
            # 2. Calcolo dei costi di trading con commissioni realistiche
            if abs(action) > 1e-6:  # Se c'è un'operazione effettiva
                self.trade_count += 1
                
                if self.trade_count <= self.free_trades_per_month:
                    # Operazione gratuita
                    trading_cost = 0
                else:
                    # Calcola il valore dell'ordine (stima basata sul cambio di posizione)
                    order_value = abs(action) * current_price
                    
                    # Calcola commissione come percentuale dell'importo
                    percentage_commission = order_value * self.commission_rate
                    
                    # Applica la commissione maggiore tra quella percentuale e quella minima
                    trading_cost = max(percentage_commission, self.min_commission)
            else:
                # Nessuna operazione, nessun costo
                trading_cost = 0

            # Calcola il delta della posizione (numero di azioni acquistate/vendute)
            trade_delta = pi_next - pi_prev

            # Aggiorna il cash: se trade_delta è positivo significa acquisto (cash diminuisce),
            # se negativo significa vendita (cash aumenta, perché - (negativo * price) è positivo)
            self.cash -= (trade_delta * current_price) + trading_cost
                        
            # 3. Penalità per dimensione della posizione (controllo del rischio)
            position_penalty = self.lambd * pi_next ** 2 * self.squared_risk
            
            # 4. Calcolo della ricompensa per allineamento con il trend
            trend_reward = 0
            # Utilizziamo le predizioni delle reti neurali se disponibili
            if all(k in self.raw_state for k in ["pred_lstm_direction", "pred_gru_direction", "pred_blstm_direction"]):
                # Calcola la direzione media prevista dalle reti
                pred_directions = [
                    self.raw_state["pred_lstm_direction"],
                    self.raw_state["pred_gru_direction"],
                    self.raw_state["pred_blstm_direction"]
                ]
                avg_direction = sum(pred_directions) / len(pred_directions)
                
                # Ricompensa se la posizione è nella stessa direzione della previsione
                if (pi_next > 0 and avg_direction > 0) or (pi_next < 0 and avg_direction < 0):
                    trend_reward = abs(pi_next) * abs(avg_direction) * 0.2
                    
            # In alternativa, possiamo usare indicatori tecnici come il MACD
            elif "MACD_Histogram" in self.raw_state:
                macd_hist = self.raw_state["MACD_Histogram"]
                if (pi_next > 0 and macd_hist > 0) or (pi_next < 0 and macd_hist < 0):
                    trend_reward = abs(pi_next) * abs(macd_hist) * 0.15
            
            # Poi, nella funzione step(), aggiungi:
            risk_penalty = self.calculate_risk(pi_next)
            # 5. Calcolo della ricompensa complessiva
            reward = (pnl - trading_cost - position_penalty - risk_penalty + trend_reward) / self.scale_reward
            
        else:
            # Usa il vecchio sistema con processo OU simulato
            self.it += 1
            self.p = self.signal[self.it + 1]
            self.done = self.it == (len(self.signal) - 2)
            
            # Vecchio calcolo della ricompensa per compatibilità
            if self.cost == "trade_0":
                reward = (self.p * pi_next - self.lambd * pi_next ** 2 * self.squared_risk - pen) / self.scale_reward
            elif self.cost == "trade_l1":
                if self.noise:
                    reward = ((self.p + self.noise_array[self.it]) * pi_next
                            - self.lambd * pi_next ** 2 * self.squared_risk
                            - self.psi * abs(pi_next - pi_prev)
                            - pen) / self.scale_reward
                else:
                    reward = (self.p * pi_next
                            - self.lambd * pi_next ** 2 * self.squared_risk
                            - self.psi * abs(pi_next - pi_prev)
                            - pen) / self.scale_reward
            elif self.cost == "trade_l2":
                if self.noise:
                    reward = ((self.p + self.noise_array[self.it]) * pi_next
                            - self.lambd * pi_next ** 2 * self.squared_risk
                            - self.psi * (pi_next - pi_prev) ** 2
                            - pen) / self.scale_reward
                else:
                    reward = (self.p * pi_next
                            - self.lambd * pi_next ** 2 * self.squared_risk
                            - self.psi * (pi_next - pi_prev) ** 2
                            - pen) / self.scale_reward
        
        # Aggiorna lo stato base per compatibilità
        self.state = (self.p, self.pi)

        return reward
    def get_portfolio_value(self):
        """
        Restituisce il valore totale del portafoglio:
        cash residuo + (numero di azioni possedute * prezzo corrente).
        """
        return self.cash + self.pi * self.p
    
    # I metodi test() e test_apply() che erano presenti nell'originale li manteniamo
    def test(
        self, agent, model, total_episodes=100, random_states=None, noise_seeds=None
    ):
        """
        Description
        ---------------
        Test a model on a number of simulated episodes and get the average cumulative
        reward.

        Parameters
        ---------------
        agent          : Agent object, the agent that loads the model.
        model          : Actor object, the actor network.
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
            - if None, do not use random state when generating episodes
              (useful to get an idea about the performance of a single model).
            - if List, generate episodes with the values in random_states (useful when
              comparing different models).

        noise_seeds    : None or List of length total_episodes:
                         - if None, do not use a random state when generating the additive
                           noise of the returns
                         - if List, generate noise with seeds in noise_seeds.

        Returns
        ---------------
        2-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        agent.actor_local = model
        if random_states is not None:
            assert total_episodes == len(
                random_states
            ), "random_states should be a list of length total_episodes!"

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                action = agent.act(state, noise=False)
                pi_next = np.clip(self.pi + action, -self.max_pos, self.max_pos)
                episode_positions.append(pi_next)
                reward = self.step(action)
                pnl = reward + (self.lambd * self.pi ** 2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = total_pnl
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #      'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )

    def apply(self, state, thresh=1, lambd=None, psi=None):
        """
        Description
        ---------------
        Apply solution with a certain band and slope outside the band, otherwise apply the
        myopic solution.

        Parameters
        ---------------
        state      : 2-tuple, the current state.
        thresh     : Float>0, price threshold to make a trade.
        lambd      : Float, slope of the solution in the non-banded region.
        psi        : Float, band width of the solution.

        Returns
        ---------------
        Float, the trade to make in state according to this function.
        """

        p, pi = state
        if lambd is None:
            lambd = self.lambd

        if psi is None:
            psi = self.psi

        if not self.squared_risk:
            if abs(p) < thresh:
                return 0
            elif p >= thresh:
                return self.max_pos - pi
            elif p <= -thresh:
                return -self.max_pos - pi

        else:
            if self.cost == "trade_0":
                return p / (2 * lambd) - pi

            elif self.cost == "trade_l2":
                return (p + 2 * psi * pi) / (2 * (lambd + psi)) - pi

            elif self.cost == "trade_l1":
                if p < -psi + 2 * lambd * pi:
                    return (p + psi) / (2 * lambd) - pi
                elif -psi + 2 * lambd * pi <= p <= psi + 2 * lambd * pi:
                    return 0
                elif p > psi + 2 * lambd * pi:
                    return (p - psi) / (2 * lambd) - pi

    def test_apply(
        self,
        total_episodes=10,
        random_states=None,
        thresh=1,
        lambd=None,
        psi=None,
        noise_seeds=None,
        max_point=6.0,
        n_points=1000,
    ):
        """
        Description
        ---------------
        Test a function with certain slope and band width for each reward model (with and
        without trading cost, and depending on the penalty when trading cost is used).
        When psi and lambd are not provided, use the myopic solution.

        Parameters
        ---------------
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
                         - if None, do not use random state when generating episodes
                           (useful to get an idea about the performance of a single
                           model).
                         - if List, generate episodes with the values in random_states
                           (useful when comparing different models).
        lambd          : Float, slope of the solution in the non-banded region.
        psi            : Float, band width of the solution.
        max_point      : Float, the maximum point in the grid [0, max_point]
        n_points       : Int, the number of points in the grid.

        Returns
        ---------------
        5-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
                  - Dict, cumulative sum of the reward at each time step per episode.
                  - Dict, pnl per episode.
                  - Dict, positions per episode.
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        if random_states is not None:
            assert total_episodes == len(
                random_states
            ), "random_states should be a list of length total_episodes!"

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                action = self.apply(state, thresh=thresh, lambd=lambd, psi=psi)
                reward = self.step(action)
                pnl = reward + (self.lambd * self.pi ** 2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                episode_positions.append(state[1])
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = episode_pnls
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #       'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )