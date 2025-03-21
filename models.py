import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    Inizializza i pesi del layer in base al numero di input (fan_in)
    usando un intervallo uniforme: (-1/sqrt(fan_in), 1/sqrt(fan_in)).
    """
    fan_in = layer.weight.data.size()[1]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    Rete neurale per la policy (Actor).

    Input:
      - state: vettore di stato di dimensione state_size (cioè, numero di feature + 1 per la posizione).

    Output:
      - Azione: un valore (per spazi d'azione continui).
    """
    def __init__(self, state_size, action_size=1, seed=0, fc1_units=128, fc2_units=64):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Inizializza i pesi dei layer.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        # Il layer finale viene inizializzato con un intervallo piccolo per output piccoli inizialmente
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.fill_(0)

    def forward(self, state):
        """
        Esegue il forward pass dell'Actor.

        Args:
            state (Tensor): input dello stato.

        Returns:
            Tensor: azione calcolata (output del network).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Critic(nn.Module):
    """
    Rete neurale per il Critic.

    Input:
      - Lo stato concatenato all'azione (dimensione = state_size + action_size).

    Output:
      - Valore Q (un singolo valore).
    """
    def __init__(self, state_size, action_size=1, seed=0, fcs1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Il primo layer prende lo stato e l'azione concatenati
        self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Inizializza i pesi dei layer del Critic.
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        # Inizializzazione del layer finale
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.fill_(0)

    def forward(self, state, action):
        """
        Esegue il forward pass del Critic.

        Args:
            state (Tensor): vettore di stato.
            action (Tensor): azione eseguita.

        Returns:
            Tensor: il valore Q della coppia (stato, azione).
        """
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)