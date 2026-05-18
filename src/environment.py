import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OptimalExecutionEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, transmat, total_to_sell=10.0, T=24):
        super(OptimalExecutionEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.transmat = transmat
        self.T = T
        self.initial_inventory = total_to_sell 
        self.inventory = total_to_sell
        
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Observation : 15 caractéristiques
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # On choisit un point de départ qui laisse assez de place pour T étapes
        high = len(self.df) - self.T - 1
        self.start_tick = np.random.randint(0, high)
        self.current_step = 0
        self.inventory = self.initial_inventory
        return self._next_observation(), {}

    def _next_observation(self):
        idx = self.start_tick + self.current_step
        row = self.df.iloc[idx]
        regime_actuel = int(row['Regime'])
        
        # Chapman-Kolmogorov
        v_t = np.zeros(3)
        v_t[regime_actuel] = 1
        prob_t_plus_1 = np.dot(v_t, self.transmat)
        A5 = np.linalg.matrix_power(self.transmat, 5)
        prob_t_plus_5 = np.dot(v_t, A5)
        
        obs = np.array([
            row['Close'] / 100000, 
            row['MACD'], 
            row['RSI'] / 100, 
            row['ADX'] / 100, 
            row['CCI'] / 100,
            regime_actuel,
            self.inventory / self.initial_inventory, # Maintenant défini !
            (self.T - self.current_step) / self.T,
            prob_t_plus_1[0], prob_t_plus_1[1], prob_t_plus_1[2],
            prob_t_plus_5[0], prob_t_plus_5[1], prob_t_plus_5[2],
            row['Volatility'] if 'Volatility' in row else 0
        ], dtype=np.float32)
        return obs

    def step(self, action):
        idx = self.start_tick + self.current_step
        price = self.df.iloc[idx]['Close']
        
        if self.current_step == self.T - 1:
            amount_to_sell = self.inventory
        else:
            amount_to_sell = action[0] * self.inventory
        
        revenue = amount_to_sell * price
        self.inventory -= amount_to_sell
        reward = revenue / 1000 
        
        self.current_step += 1
        done = (self.current_step >= self.T) or (self.inventory < 0.0001)
        
        if done and self.inventory > 0.001:
            reward -= (self.inventory * price * 0.1) / 1000
            
        return self._next_observation(), reward, done, False, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Inv: {self.inventory:.4f}")