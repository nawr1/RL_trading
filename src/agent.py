import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from data_processor import add_technical_indicators, extract_regimes
from environment import OptimalExecutionEnv

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_FILE   = os.path.join("data", "BTCUSDT_1h_FINAL.csv")
REGIME_FILE = os.path.join("data", "BTCUSDT_1h_REGIMES.csv")
MODEL_DIR   = "models"
LOG_DIR     = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ─── Chapman–Kolmogorov Constraint Callback ───────────────────────────────────

class CKConstraintCallback(BaseCallback):
    """
    At each rollout end, log the KL divergence between the empirical
    regime-transition frequencies visited during rollout and the theoretical
    Chapman–Kolmogorov transition matrix.  Optionally add a penalty to the
    value loss when the constraint is violated.
    """

    def __init__(self, transmat: np.ndarray, threshold: float = 0.05,
                 verbose: int = 0):
        super().__init__(verbose)
        self.transmat  = transmat          # theoretical CK matrix
        self.threshold = threshold         # KL threshold before logging warning
        self._trans_counts = np.zeros_like(transmat)
        self._prev_regime  = None

    def _on_step(self) -> bool:
        # Extract current regime from observation (index 5)
        obs   = self.locals.get("new_obs", None)
        if obs is None:
            return True
        regime = int(np.round(obs[0][5]))  # obs[0] because VecEnv
        if self._prev_regime is not None:
            r = min(self._prev_regime, self._trans_counts.shape[0] - 1)
            c = min(regime,            self._trans_counts.shape[1] - 1)
            self._trans_counts[r, c] += 1
        self._prev_regime = regime
        return True

    def _on_rollout_end(self) -> None:
        row_sums = self._trans_counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            emp = np.where(row_sums > 0,
                           self._trans_counts / row_sums,
                           1.0 / self._trans_counts.shape[1])
        # Symmetric KL divergence
        kl = 0.5 * np.sum(
            emp * np.log((emp + 1e-9) / (self.transmat + 1e-9)) +
            self.transmat * np.log((self.transmat + 1e-9) / (emp + 1e-9))
        )
        self.logger.record("ck/kl_divergence", float(kl))
        if kl > self.threshold:
            self.logger.record("ck/constraint_violated", 1.0)
            if self.verbose:
                print(f"  [CK] KL={kl:.4f} > threshold={self.threshold}")
        else:
            self.logger.record("ck/constraint_violated", 0.0)
        # Reset counts
        self._trans_counts[:] = 0
        self._prev_regime = None


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    if os.path.exists(REGIME_FILE):
        print("Loading pre-computed regime data …")
        df = pd.read_csv(REGIME_FILE, parse_dates=['Date'])
        df = df.set_index('Date')
        transmat = np.load(os.path.join("data", "transmat.npy"))
    else:
        print("Computing features & regimes …")
        df_raw = pd.read_csv(DATA_FILE, parse_dates=['Open_Time'])
        df_raw = df_raw.rename(columns={'Open_Time': 'Date'}).set_index('Date')
        df_feat = add_technical_indicators(df_raw)
        df, transmat, _, _ = extract_regimes(df_feat, n_regimes=3)
        df.to_csv(REGIME_FILE)
        np.save(os.path.join("data", "transmat.npy"), transmat)
    return df.reset_index(), transmat


def make_env(df, transmat, T=24, total_to_sell=10.0):
    def _init():
        env = OptimalExecutionEnv(df, transmat,
                                  total_to_sell=total_to_sell, T=T)
        env = Monitor(env)
        return env
    return _init


def train(
    total_timesteps: int = 500_000,
    T: int = 24,
    total_to_sell: float = 10.0,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    batch_size: int = 512,
    n_epochs: int = 10,
):
    df, transmat = load_data()

    split = int(len(df) * 0.8)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_eval  = df.iloc[split:].reset_index(drop=True)
    print(f"Train rows: {len(df_train):,}  |  Eval rows: {len(df_eval):,}")

    train_envs = DummyVecEnv([make_env(df_train, transmat, T, total_to_sell)
                               for _ in range(n_envs)])
    train_envs = VecNormalize(train_envs, norm_obs=True, norm_reward=True,
                              clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(df_eval, transmat, T, total_to_sell)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)


    policy_kwargs = dict(net_arch=[256, 256, 128])

    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    ck_cb = CKConstraintCallback(transmat, threshold=0.05, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_execution",
        verbose=1,
    )

    print(f"\nStarting PPO training for {total_timesteps:,} steps …")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[ck_cb, eval_cb, ckpt_cb],
        tb_log_name="ppo_ck_execution",
        progress_bar=True,
    )
    model_path = os.path.join(MODEL_DIR, "ppo_final")
    model.save(model_path)
    train_envs.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print(f"\nModel saved → {model_path}.zip")
    print(f"VecNormalize saved → {MODEL_DIR}/vec_normalize.pkl")

    return model, train_envs, df_train, df_eval, transmat

if __name__ == "__main__":
    train(total_timesteps=500_000)