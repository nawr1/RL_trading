import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class OptimalExecutionEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    # ── Market-impact / penalty hyper-params ──────────────────────────────────
    IMPACT_ETA   = 0.002
    URGENCY_COEF = 0.30
    TWAP_BONUS   = 0.15
    REGIME_COEF  = 0.05

    def __init__(
        self,
        df: pd.DataFrame,
        transmat: np.ndarray,
        total_to_sell: float = 10.0,
        T: int = 24,
        reward_scale: float = 1_000.0,
    ):
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.transmat        = np.array(transmat, dtype=np.float64)
        self.T               = T
        self.initial_inventory = total_to_sell
        self.reward_scale    = reward_scale

        self._A5 = np.linalg.matrix_power(self.transmat, 5)

        self.action_space      = spaces.Box(low=0.0, high=1.0,
                                            shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(15,), dtype=np.float32)
        self.current_step  = 0
        self.start_tick    = 0
        self.inventory     = self.initial_inventory

    def _ck_probs(self, regime: int):
        e_t = np.zeros(len(self.transmat))
        e_t[regime] = 1.0
        p1 = e_t @ self.transmat
        p5 = e_t @ self._A5
        return p1, p5

    def _next_observation(self) -> np.ndarray:
        # Clamp index pour éviter out-of-bounds en fin d'épisode
        idx = min(self.start_tick + self.current_step, len(self.df) - 1)
        row = self.df.iloc[idx]
        regime = int(row['Regime'])
        p1, p5 = self._ck_probs(regime)

        obs = np.array([
            row['Close'] / 100_000,
            row['MACD'],
            row['RSI']  / 100,
            row['ADX']  / 100,
            row['CCI']  / 100,
            float(regime),
            self.inventory / self.initial_inventory,
            (self.T - self.current_step) / self.T,
            p1[0], p1[1], p1[2],
            p5[0], p5[1], p5[2],
            float(row.get('Volatility', 0.0)),
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        high = max(1, len(self.df) - self.T - 1)
        self.start_tick   = int(self.np_random.integers(0, high))
        self.current_step = 0
        self.inventory    = self.initial_inventory
        return self._next_observation(), {}

    def step(self, action):
        # 1. Données marché
        idx   = self.start_tick + self.current_step
        row   = self.df.iloc[idx]
        price = float(row['Close'])

        # 2. Action robuste
        action_val = np.reshape(np.asarray(action), -1)[0]

        # 3. Quantité à vendre
        if self.current_step == self.T - 1:
            amount = self.inventory
        else:
            amount = float(np.clip(action_val, 0.0, 1.0)) * self.inventory
        amount = max(amount, 0.0)

        # 4. Impact de marché quadratique
        impact_ratio    = amount / (self.initial_inventory + 1e-9)
        effective_price = price * (1.0 - self.IMPACT_ETA * impact_ratio ** 2)
        revenue         = amount * effective_price

        # 5. Reward de base
        reward = revenue / self.reward_scale

        # 6. Bonus vs TWAP
        twap_ref = price
        if amount > 1e-4 and effective_price > twap_ref * 0.995:
            reward += self.TWAP_BONUS * (effective_price - twap_ref * 0.995) * amount / self.reward_scale

        # 7. Bonus/Malus régime HMM
        regime    = int(row['Regime'])
        time_left = (self.T - self.current_step) / self.T
        if regime == 0:
            reward += self.REGIME_COEF * amount / (self.initial_inventory + 1e-9)
        elif regime == 2 and time_left > 0.3:
            reward += self.REGIME_COEF * 0.5 * (1 - impact_ratio)

        # 8. Mise à jour inventaire
        self.inventory    -= amount
        self.current_step += 1
        done = (self.current_step >= self.T) or (self.inventory < 1e-4)

        # 9. Pénalité urgence
        if done and self.inventory > 1e-3:
            penalty = (self.inventory * price * self.URGENCY_COEF) / self.reward_scale
            reward -= penalty
        obs = self._next_observation()

        info = {
            'price':     price,
            'amount':    amount,
            'revenue':   revenue,
            'inventory': self.inventory,
            'regime':    regime,
        }

        return obs, reward, done, False, info

    def render(self, mode='human'):
        idx = min(self.start_tick + self.current_step - 1, len(self.df) - 1)
        price = float(self.df.iloc[idx]['Close'])
        print(f"Step {self.current_step:>3d}/{self.T} | "
              f"Inv: {self.inventory:.4f} | "
              f"Price: {price:,.1f}")