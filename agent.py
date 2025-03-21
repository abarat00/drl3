import os
from time import sleep
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

# Import dei moduli locali
from memory import Memory, PrioritizedMemory, Node
from models import Actor, Critic

# Definizione di un namedtuple per le transizioni
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "dones"))

# Parametri globali
GAMMA = 0.99                 # discount factor
TAU_ACTOR = 1e-1             # soft update parameter for the actor target network
TAU_CRITIC = 1e-3            # soft update parameter for the critic target network
LR_ACTOR = 1e-4              # learning rate for the actor network
LR_CRITIC = 1e-3             # learning rate for the critic network
WEIGHT_DECAY_actor = 0       # L2 weight decay for actor
WEIGHT_DECAY_critic = 1e-2   # L2 weight decay for critic
BATCH_SIZE = 64              # minibatch size
BUFFER_SIZE = int(1e6)       # replay buffer size
PRETRAIN = 64                # number of pretraining steps (should be > BATCH_SIZE)
MAX_STEP = 100               # number of steps in an episode
WEIGHTS = "weights/"         # path where to save model weights

# Dimensioni dei layer nelle reti originali (possono essere modificate)
FC1_UNITS_ACTOR = 16         # actor: nodes in first hidden layer (originale)
FC2_UNITS_ACTOR = 8          # actor: nodes in second hidden layer (originale)
# Le modifiche per il nuovo input prevedono architetture più grandi:
NEW_FC1_UNITS_ACTOR = 128    # actor: nuovi layer con più neuroni
NEW_FC2_UNITS_ACTOR = 64

FC1_UNITS_CRITIC = 64        # critic: nodes in first hidden layer (originale)
FC2_UNITS_CRITIC = 32        # critic: nodes in second hidden layer (originale)
NEW_FCS1_UNITS_CRITIC = 256   # critic: nuovo primo layer (stato+azione)
NEW_FC2_UNITS_CRITIC = 128    # critic: nuovo secondo layer

DECAY_RATE = 0               # decay rate for exploration noise
EXPLORE_STOP = 1e-3          # final exploration probability

def optimal_f(p, pi, lambd=0.5, psi=0.3, cost="trade_l2"):
    """
    Computes the optimal trade for different cost models.

    p     : next signal value
    pi    : current position
    lambd : cost model parameter
    psi   : trading cost parameter
    cost  : 'trade_0', 'trade_l1' or 'trade_l2'

    Returns the optimal trade.
    """
    if cost == "trade_0":
        return p / (2 * lambd) - pi
    elif cost == "trade_l2":
        return p / (2 * (lambd + psi)) + psi * pi / (lambd + psi) - pi
    elif cost == "trade_l1":
        if p <= -psi + 2 * lambd * pi:
            return (p + psi) / (2 * lambd) - pi
        elif -psi + 2 * lambd * pi < p < psi + 2 * lambd * pi:
            return 0
        elif p >= psi + 2 * lambd * pi:
            return (p - psi) / (2 * lambd) - pi

def optimal_max_pos(p, pi, thresh, max_pos):
    """
    Computes the optimal trade for MaxPos cost model with l1 trading cost.
    """
    if abs(p) < thresh:
        return 0
    elif p >= thresh:
        return max_pos - pi
    elif p <= -thresh:
        return -max_pos - pi

# Vectorized versions for efficiency
optimal_f_vec = np.vectorize(optimal_f, excluded=set(["pi", "lambd", "psi", "cost"]))
optimal_max_pos_vec = np.vectorize(optimal_max_pos, excluded=set(["pi", "thresh", "max_pos"]))

class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated exploration noise.
    """
    def __init__(self, mu=0.0, theta=0.1, sigma=0.1):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self, truncate=False, max_pos=2, position=0, action=0):
        x = self.state
        if truncate:
            from scipy.stats import truncnorm
            m = -max_pos - position - action - (1 - self.theta) * x
            M = max_pos - position - action - (1 - self.theta) * x
            x_a, x_b = m / self.sigma, M / self.sigma
            X = truncnorm(x_a, x_b, scale=self.sigma)
            dx = self.theta * (self.mu - x) + X.rvs()
            self.state = x + dx
            return self.state
        else:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn()
            self.state = x + dx
            return self.state

class Agent:
    def __init__(
        self,
        gamma=GAMMA,
        max_size=BUFFER_SIZE,
        max_step=MAX_STEP,
        memory_type="uniform",
        alpha=0.6,
        beta0=0.4,
        epsilon=1e-8,
        sliding="oldest",
        batch_size=BATCH_SIZE,
        theta=1.0,
        sigma=1.0,
    ):
        """
        Constructor of the DDPG Agent.

        Parameters:
          - gamma: discount factor.
          - max_size: maximum replay buffer size.
          - max_step: number of steps per episode.
          - memory_type: 'uniform' or 'prioritized'.
          - alpha, beta0, epsilon: parameters for prioritized replay.
          - sliding: strategy for replacement in prioritized replay.
          - batch_size: training batch size.
          - theta, sigma: parameters for the OU exploration noise.
        """
        assert 0 <= gamma <= 1, "Gamma must be in [0,1]"
        assert memory_type in ["uniform", "prioritized", "per_intervals"], "Invalid memory type"
        self.gamma = gamma
        self.max_size = max_size
        self.memory_type = memory_type
        self.epsilon = epsilon

        if memory_type == "uniform":
            self.memory = Memory(max_size=max_size)
        elif memory_type == "prioritized":
            self.memory = PrioritizedMemory(max_size=max_size, sliding=sliding)

        self.max_step = max_step
        self.alpha = alpha
        self.beta0 = beta0
        self.batch_size = batch_size
        self.noise = OUNoise(theta=theta, sigma=sigma)

        # Initialize networks (will be created during training)
        self.actor_local = None
        self.actor_target = None
        self.critic_local = None
        self.critic_target = None

    def reset(self):
        """
        Resets the exploration noise.
        """
        self.noise.reset()

    def step(self, state, action, reward, next_state, done, pretrain=False):
        """
        Saves a transition (state, action, reward, next_state, not done) in the replay buffer.
        """
        state_mb = torch.tensor([state], dtype=torch.float)
        action_mb = torch.tensor([[action]], dtype=torch.float)
        reward_mb = torch.tensor([[reward]], dtype=torch.float)
        next_state_mb = torch.tensor([next_state], dtype=torch.float)
        not_done_mb = torch.tensor([[not done]], dtype=torch.float)

        if self.memory_type == "uniform":
            self.memory.add((state_mb, action_mb, reward_mb, next_state_mb, not_done_mb))
        elif self.memory_type == "prioritized":
            priority = (abs(reward) + self.epsilon) ** self.alpha if pretrain else self.memory.highest_priority()
            self.memory.add((state_mb, action_mb, reward_mb, next_state_mb, not_done_mb), priority)

    def act(self, state, noise=True, explore_probability=1, truncate=False, max_pos=2):
        """
        Returns an action for a given state using the actor network, with optional exploration noise.

        Parameters:
          - state: the current state (expected to be a NumPy array of dimension state_size).
          - noise: whether to add exploration noise.
          - explore_probability: scaling factor for the noise.
          - truncate: if True, truncate the noise so that the resulting position stays within bounds.
          - max_pos: maximum allowed position.

        Returns:
          - action: a float representing the trade.
        """
        # In our new state, the position is the last element.
        position = state[-1]
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # per aggiungere la dimensione del batch
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_tensor).data.numpy()
        self.actor_local.train()
        if noise:
            noise_sample = self.noise.sample(truncate=truncate, max_pos=max_pos, position=position, action=float(action))
            action += explore_probability * noise_sample
        return float(action)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft-update model parameters: target = tau*local + (1-tau)*target.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def pretrain(self, env, total_steps=PRETRAIN):
        """
        Pretrains the agent by filling the replay buffer.
        """
        env.reset()
        with torch.no_grad():
            for i in range(total_steps):
                state = env.get_state()
                action = self.act(state, truncate=(not env.squared_risk), max_pos=env.max_pos)
                reward = env.step(action)
                next_state = env.get_state()
                done = env.done
                self.step(state, action, reward, next_state, done, pretrain=True)
                if done:
                    env.reset()

    def train(
        self,
        env,
        total_episodes=100,
        tau_actor=TAU_ACTOR,
        tau_critic=TAU_CRITIC,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        weight_decay_actor=WEIGHT_DECAY_actor,
        weight_decay_critic=WEIGHT_DECAY_critic,
        total_steps=PRETRAIN,
        weights=WEIGHTS,
        freq=50,
        fc1_units_actor=NEW_FC1_UNITS_ACTOR,
        fc2_units_actor=NEW_FC2_UNITS_ACTOR,
        fc1_units_critic=NEW_FCS1_UNITS_CRITIC,
        fc2_units_critic=NEW_FC2_UNITS_CRITIC,
        decay_rate=DECAY_RATE,
        explore_stop=EXPLORE_STOP,
        tensordir="runs/",
        learn_freq=50,
        plots=False,
        pi=0.5,
        lambd=None,
        psi=None,
        phi=None,
        thresh=3,
        mile=50,
        progress="tqdm_notebook",
    ):
        """
        Trains the agent over a number of episodes.
        """
        if not os.path.isdir(weights):
            os.mkdir(weights)

        writer = SummaryWriter(log_dir=tensordir)

        # Optional plotting setup.
        if plots:
            plt.figure(figsize=(15, 10))
            range_values = np.arange(-4, 4, 0.01)
            signal_zeros = torch.tensor(np.vstack((range_values, np.zeros(len(range_values)))).T, dtype=torch.float)
            signal_ones_pos = torch.tensor(np.vstack((range_values, 0.5 * np.ones(len(range_values)))).T, dtype=torch.float)
            signal_ones_neg = torch.tensor(np.vstack((range_values, -0.5 * np.ones(len(range_values)))).T, dtype=torch.float)
            if psi is None:
                psi = env.psi
            if lambd is None:
                lambd = env.lambd
            if env.squared_risk:
                result1 = optimal_f_vec(signal_ones_neg[:, 0].numpy(), -pi, lambd=lambd, psi=psi, cost=env.cost)
                result2 = optimal_f_vec(signal_zeros[:, 0].numpy(), 0, lambd=lambd, psi=psi, cost=env.cost)
                result3 = optimal_f_vec(signal_ones_pos[:, 0].numpy(), pi, lambd=lambd, psi=psi, cost=env.cost)
            else:
                result1 = optimal_max_pos_vec(signal_ones_neg[:, 0].numpy(), -pi, thresh, env.max_pos)
                result2 = optimal_max_pos_vec(signal_zeros[:, 0].numpy(), 0, thresh, env.max_pos)
                result3 = optimal_max_pos_vec(signal_ones_pos[:, 0].numpy(), pi, thresh, env.max_pos)

        # Initialize Actor networks
        self.actor_local = Actor(env.state_size, fc1_units=fc1_units_actor, fc2_units=fc2_units_actor)
        self.actor_target = Actor(env.state_size, fc1_units=fc1_units_actor, fc2_units=fc2_units_actor)
        actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor, weight_decay=weight_decay_actor)
        actor_lr_scheduler = lr_scheduler.StepLR(actor_optimizer, step_size=mile * 100, gamma=0.5)

        # Initialize Critic networks
        self.critic_local = Critic(env.state_size, fcs1_units=fc1_units_critic, fc2_units=fc2_units_critic)
        self.critic_target = Critic(env.state_size, fcs1_units=fc1_units_critic, fc2_units=fc2_units_critic)
        critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)
        critic_lr_scheduler = lr_scheduler.StepLR(critic_optimizer, step_size=mile * 100, gamma=0.5)

        # Save the initialized Actor network
        model_file = weights + "ddpg1" + ".pth"
        torch.save(self.actor_local.state_dict(), model_file)

        mean_rewards = deque(maxlen=10)
        cum_rewards = []
        actor_losses = deque(maxlen=10)
        critic_losses = deque(maxlen=10)

        Node.reset_count()
        self.pretrain(env, total_steps=total_steps)
        i = 0
        N_train = total_episodes * env.T // learn_freq
        beta = self.beta0
        self.reset()
        n_train = 0

        range_total_episodes = list(range(total_episodes))
        if progress == "tqdm_notebook":
            from tqdm import tqdm_notebook
            range_total_episodes = tqdm_notebook(range_total_episodes)
            progress_bar = range_total_episodes
        elif progress == "tqdm":
            from tqdm import tqdm
            range_total_episodes = tqdm(range_total_episodes)
            progress_bar = range_total_episodes
        else:
            progress_bar = None

        for episode in range_total_episodes:
            episode_rewards = []
            env.reset()
            state = env.get_state()
            done = env.done
            train_iter = 0

            while not done:
                explore_probability = explore_stop + (1 - explore_stop) * np.exp(-decay_rate * i)
                action = self.act(state, truncate=(not env.squared_risk), max_pos=env.max_pos, explore_probability=explore_probability)
                reward = env.step(action)
                writer.add_scalar("State/signal", state[0], i)
                writer.add_scalar("Signal/position", state[-1], i)  # usa state[-1] per la posizione
                writer.add_scalar("Signal/action", action, i)
                next_state = env.get_state()
                done = env.done
                self.step(state, action, reward, next_state, done)
                state = next_state
                episode_rewards.append(reward)
                i += 1
                train_iter += 1
                if done:
                    self.reset()
                    total_reward = np.sum(episode_rewards)
                    mean_rewards.append(total_reward)
                    if (episode > 0) and (episode % 5 == 0):
                        mean_r = np.mean(mean_rewards)
                        cum_rewards.append(mean_r)
                        writer.add_scalar("Reward & Loss/reward", mean_r, episode)
                        writer.add_scalar("Reward & Loss/actor_loss", np.mean(actor_losses), episode)
                        writer.add_scalar("Reward & Loss/critic_loss", np.mean(critic_losses), episode)

                if train_iter % learn_freq == 0:
                    n_train += 1
                    if self.memory_type == "uniform":
                        transitions = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)
                    elif self.memory_type == "prioritized":
                        transitions, indices = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)

                    actions_next = self.actor_target(next_states_mb)
                    Q_targets_next = self.critic_target(next_states_mb, actions_next)
                    Q_targets = rewards_mb + (self.gamma * Q_targets_next * dones_mb)
                    Q_expected = self.critic_local(states_mb, actions_mb)
                    td_errors = F.l1_loss(Q_expected, Q_targets, reduction="none")
                    if self.memory_type == "prioritized":
                        sum_priorities = self.memory.sum_priorities()
                        probabilities = (self.memory.retrieve_priorities(indices) / sum_priorities).reshape((-1, 1))
                        is_weights = torch.tensor(1 / ((self.max_size * probabilities) ** beta), dtype=torch.float)
                        is_weights /= is_weights.max()
                        beta = (1 - self.beta0) * (n_train / N_train) + self.beta0
                        for i_enum, index in enumerate(indices):
                            self.memory.update(index, (abs(float(td_errors[i_enum].data)) + self.epsilon) ** self.alpha)
                        critic_loss = (is_weights * (td_errors ** 2)).mean() / 2
                    elif self.memory_type == "uniform":
                        critic_loss = (td_errors ** 2).mean() / 2

                    critic_losses.append(critic_loss.data.item())
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 0.1)
                    critic_optimizer.step()
                    critic_lr_scheduler.step()

                    actions_pred = self.actor_local(states_mb)
                    actor_loss = -self.critic_local(states_mb, actions_pred).mean()
                    actor_losses.append(actor_loss.data.item())
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 0.1)
                    actor_optimizer.step()
                    actor_lr_scheduler.step()

                    self.soft_update(self.critic_local, self.critic_target, tau_critic)
                    self.soft_update(self.actor_local, self.actor_target, tau_actor)

            if plots:
                plt.clf()
                self.actor_local.eval()
                with torch.no_grad():
                    plt.subplot(2, 3, 1)
                    plt.plot(signal_ones_neg[:, 0].numpy(), self.actor_local(signal_ones_neg)[:, 0].data.numpy(), label="model")
                    plt.plot(signal_ones_neg[:, 0].numpy(), result1, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 2)
                    plt.plot(signal_zeros[:, 0].numpy(), self.actor_local(signal_zeros)[:, 0].data.numpy(), label="model")
                    plt.plot(signal_zeros[:, 0].numpy(), result2, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 3)
                    plt.plot(signal_ones_pos[:, 0].numpy(), self.actor_local(signal_ones_pos)[:, 0].data.numpy(), label="model")
                    plt.plot(signal_ones_pos[:, 0].numpy(), result3, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 4)
                    sns.distplot(states_mb[:, 0])
                display.clear_output(wait=True)
                if progress_bar is not None:
                    display.display(progress_bar)
                display.display(plt.gcf())
                sleep(0.0001)
                self.actor_local.train()

            if (episode % freq) == 0:
                model_file = weights + "ddpg" + str(episode) + ".pth"
                torch.save(self.actor_local.state_dict(), model_file)

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()