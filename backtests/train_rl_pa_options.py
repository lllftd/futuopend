import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import time

# Add the project root to sys.path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from core.utils import load_price_data
from core.pa_rules import add_all_pa_features
from core.indicators import atr, add_pseudo_cvd, cvd_divergence_features

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. 数据准备函数
# ---------------------------------------------------------------------------
def prepare_rl_data(symbol="SPY", start_date="2024-01-01", end_date="2026-01-01"):
    print(f"Loading data for {symbol}...")
    raw = load_price_data(symbol)
    raw["time_key"] = pd.to_datetime(raw["time_key"])
    raw = raw[(raw["time_key"] >= start_date) & (raw["time_key"] < end_date)].reset_index(drop=True)

    raw_idx = raw.set_index("time_key")
    df_5m = raw_idx[["open", "high", "low", "close", "volume"]].resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"]).reset_index()

    df_5m["atr_raw"] = atr(df_5m, length=14)
    df_5m["ema_20"] = df_5m["close"].ewm(span=20, adjust=False).mean()

    print("Computing PA features...")
    df_pa = add_all_pa_features(df_5m, df_5m["atr_raw"], timeframe="1min")

    rng_5m = np.maximum(df_5m["high"] - df_5m["low"], 1e-12)
    body_5m = np.abs(df_5m["close"] - df_5m["open"])
    df_5m["body_ratio"] = body_5m / rng_5m
    df_5m["body_mult"] = (body_5m / body_5m.rolling(20, min_periods=5).mean()).fillna(0)
    ema_slope_5m = df_5m["ema_20"].diff()
    ema_slope_abs_ma40_5m = ema_slope_5m.abs().rolling(40, min_periods=10).mean().replace(0, np.nan)
    df_5m["ema_slope_ratio"] = (ema_slope_5m / ema_slope_abs_ma40_5m).fillna(0)

    df_pa["prev_5m_body_ratio"] = df_5m["body_ratio"]
    df_pa["prev_5m_body_mult"] = df_5m["body_mult"]
    df_pa["prev_5m_ema_slope_ratio"] = df_5m["ema_slope_ratio"]

    df_pa["signal_time"] = df_pa["time_key"] + pd.Timedelta(minutes=5)
    df_pa = df_pa.rename(columns={
        "high": "prev_5m_high",
        "low": "prev_5m_low",
        "close": "prev_5m_close",
        "open": "prev_5m_open",
        "atr_raw": "prev_5m_atr",
        "ema_20": "prev_5m_ema20"
    })
    df_pa = df_pa.drop(columns=["time_key", "volume"])

    df_1m = pd.merge_asof(
        raw.sort_values("time_key"),
        df_pa.rename(columns={"signal_time": "time_key"}).sort_values("time_key"),
        on="time_key",
        direction="backward",
        tolerance=pd.Timedelta(minutes=4)
    )
    df_1m = df_1m.dropna(subset=["pa_env_state"]).reset_index(drop=True)

    print("Computing CVD features...")
    cvd_df = add_pseudo_cvd(df_1m, method="clv_body_volume")[["cvd_pressure", "cvd_session"]]
    df_1m = df_1m.join(cvd_df)
    div_df = cvd_divergence_features(df_1m, cvd_column="cvd_session", lookback=10, slope_lookback=3)
    df_1m = df_1m.join(div_df)

    # --- 归一化特征 (Normalization for RL) ---
    print("Normalizing features for RL...")
    df_1m['pa_env_score_total_norm'] = df_1m['pa_env_score_total'] / 10.0
    df_1m['pa_net_pressure_norm'] = df_1m['pa_net_pressure'] / 100.0
    df_1m['prev_5m_body_ratio_norm'] = df_1m['prev_5m_body_ratio']
    df_1m['prev_5m_body_mult_norm'] = np.clip(df_1m['prev_5m_body_mult'] / 5.0, 0, 1)
    df_1m['prev_5m_ema_slope_ratio_norm'] = np.clip(df_1m['prev_5m_ema_slope_ratio'] / 5.0, -1, 1)
    df_1m['cvd_pressure_norm'] = np.clip(df_1m['cvd_pressure'] / 100.0, -1, 1)
    
    minutes_from_open = (df_1m['time_key'].dt.hour - 9) * 60 + df_1m['time_key'].dt.minute - 30
    df_1m['time_of_day_norm'] = np.clip(minutes_from_open / 390.0, 0, 1)

    # New continuous features normalization
    df_1m['pa_buy_signal_score_norm'] = df_1m['pa_buy_signal_score'].fillna(0) / 7.0
    df_1m['pa_sell_signal_score_norm'] = df_1m['pa_sell_signal_score'].fillna(0) / 7.0
    df_1m['pa_sr_position_norm'] = df_1m['pa_sr_position'].fillna(0.5)
    df_1m['pa_round_number_dist_norm'] = np.clip(df_1m['pa_round_number_dist'].fillna(1.0) / 3.0, 0, 1)
    df_1m['pa_mtr_score_norm'] = df_1m['pa_mtr_score'].fillna(0) / 3.0
    df_1m['pa_consec_climax_count_norm'] = np.clip(df_1m['pa_consec_climax_count'].fillna(0) / 5.0, 0, 1)
    df_1m['pa_momentum_accel_norm'] = np.clip(df_1m['pa_momentum_accel'].fillna(0), -1, 1)
    df_1m['pa_follow_through_score_up_norm'] = df_1m['pa_follow_through_score_up'].fillna(0) / 5.0
    df_1m['pa_follow_through_score_down_norm'] = df_1m['pa_follow_through_score_down'].fillna(0) / 5.0
    df_1m['pa_counter_trend_quality_long_norm'] = df_1m['pa_counter_trend_quality_long'].fillna(0) / 7.0
    df_1m['pa_counter_trend_quality_short_norm'] = df_1m['pa_counter_trend_quality_short'].fillna(0) / 7.0
    df_1m['pa_channel_position_norm'] = df_1m['pa_channel_position'].fillna(0.5)
    df_1m['pa_trend_resume_prob_norm'] = df_1m['pa_trend_resume_prob'].fillna(0.8)
    df_1m['pa_triangle_breakout_bias_norm'] = df_1m['pa_triangle_breakout_bias'].fillna(0)
    df_1m['pa_session_phase_norm'] = df_1m['pa_session_phase'].fillna(0.5)
    df_1m['pa_cycle_phase_norm'] = df_1m['pa_cycle_phase'].fillna(0) / 3.0
    df_1m['pa_cycle_age_norm'] = np.clip(df_1m['pa_cycle_age'].fillna(0) / 50.0, 0, 1)

    # New batch 3 continuous features
    df_1m['pa_prev_day_dir_norm'] = df_1m['pa_prev_day_dir'].fillna(0)
    df_1m['pa_prev_day_range_pct_norm'] = np.clip(df_1m['pa_prev_day_range_pct'].fillna(0) / 0.05, 0, 1)

    bool_cols = [
        'pa_is_h2_setup', 'pa_is_l2_setup', 'pa_breakout_success_up', 'pa_breakout_success_down',
        'pa_wedge_third_push_up', 'pa_wedge_third_push_down', 'pa_channel_overshoot_revert_up',
        'pa_channel_overshoot_revert_down', 'pa_mag20_bull', 'pa_mag20_bear',
        # New boolean features
        'pa_is_strong_buy_signal', 'pa_is_strong_sell_signal',
        'pa_is_outside_bar', 'pa_is_engulfing_bull', 'pa_is_engulfing_bear',
        'pa_multi_bar_reversal_bull', 'pa_multi_bar_reversal_bear',
        'pa_at_50pct_retrace',
        'pa_double_top', 'pa_double_bottom', 'pa_dt_is_flag', 'pa_db_is_flag',
        'pa_mtr_bull_ready', 'pa_mtr_bear_ready',
        'pa_buy_climax', 'pa_sell_climax', 'pa_v_shape_top', 'pa_v_shape_bottom',
        'pa_final_flag_bull', 'pa_final_flag_bear',
        'pa_tbtl_correction_up', 'pa_tbtl_correction_down',
        'pa_bull_body_growing', 'pa_bear_body_growing',
        'pa_bull_body_shrinking', 'pa_bear_body_shrinking',
        'pa_at_channel_buy_zone', 'pa_at_channel_sell_zone',
        'pa_endless_pullback',
        'pa_island_reversal_top', 'pa_island_reversal_bottom', 'pa_negative_gap',
        'pa_triangle_converging', 'pa_triangle_expanding',
        # Batch 3 boolean features
        'pa_breakout_volume_confirm', 'pa_breakout_volume_missing',
        'pa_head_shoulders_top', 'pa_head_shoulders_bottom',
        'pa_parabolic_wedge_up', 'pa_parabolic_wedge_down',
        'pa_prev_day_hl_fail_up', 'pa_prev_day_hl_fail_down',
        'pa_opening_reversal', 'pa_gap_open_flag',
    ]
    for c in bool_cols:
        df_1m[c + "_norm"] = df_1m[c].fillna(False).astype(float)

    # New continuous normalized feature names
    new_cont_norm_cols = [
        'pa_buy_signal_score_norm', 'pa_sell_signal_score_norm',
        'pa_sr_position_norm', 'pa_round_number_dist_norm',
        'pa_mtr_score_norm', 'pa_consec_climax_count_norm',
        'pa_momentum_accel_norm',
        'pa_follow_through_score_up_norm', 'pa_follow_through_score_down_norm',
        'pa_counter_trend_quality_long_norm', 'pa_counter_trend_quality_short_norm',
        'pa_channel_position_norm', 'pa_trend_resume_prob_norm',
        'pa_triangle_breakout_bias_norm',
        'pa_session_phase_norm', 'pa_cycle_phase_norm', 'pa_cycle_age_norm',
        'pa_prev_day_dir_norm', 'pa_prev_day_range_pct_norm',
    ]

    feature_cols = [
        'pa_env_score_total_norm', 'pa_net_pressure_norm', 'prev_5m_body_ratio_norm',
        'prev_5m_body_mult_norm', 'prev_5m_ema_slope_ratio_norm', 'cvd_pressure_norm',
        'time_of_day_norm'
    ] + [c + "_norm" for c in bool_cols] + new_cont_norm_cols

    # 按天分组
    days = [group for _, group in df_1m.groupby(df_1m['time_key'].dt.date)]
    print(f"Prepared {len(days)} trading days of data.")
    return days, feature_cols


# ---------------------------------------------------------------------------
# 2. 自定义 RL 环境 (Options PA 1m)
# ---------------------------------------------------------------------------
class PAOptionsDayEnv(gym.Env):
    """
    1分钟级别的 PA 期权交易环境。
    强调高盈亏比和爆发力，引入 Theta 惩罚机制。
    每次 Episode 为一个完整的交易日。
    """
    def __init__(self, days_data, feature_cols, cost=0.0002, sl_atr=1.0, tp_atr=3.0, max_holding_bars=45):
        super().__init__()
        self.days_data = days_data
        self.feature_cols = feature_cols
        
        # 交易设定
        self.cost = cost  # 手续费/滑点
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.max_holding_bars = max_holding_bars  # 时间止损
        
        # 动作空间：0 = 观望, 1 = 做多, 2 = 做空
        self.action_space = spaces.Discrete(3)
        
        # 观测空间：只看当前的环境特征，因为 RL 只有在“空仓”时才做决策
        obs_dim = len(self.feature_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day_idx = np.random.randint(len(self.days_data))
        self.day_df = self.days_data[self.current_day_idx].reset_index(drop=True)
        
        self.current_step = 0
        
        self.features_array = self.day_df[self.feature_cols].to_numpy(dtype=np.float32)
        self.close_prices = self.day_df['close'].to_numpy(dtype=np.float32)
        self.high_prices = self.day_df['high'].to_numpy(dtype=np.float32)
        self.low_prices = self.day_df['low'].to_numpy(dtype=np.float32)
        self.open_prices = self.day_df['open'].to_numpy(dtype=np.float32)
        self.times = self.day_df['time_key'].dt.time.values
        self.atr_array = self.day_df['prev_5m_atr'].to_numpy(dtype=np.float32)
        
        return self._get_obs(), {}

    def _get_obs(self):
        return self.features_array[self.current_step]

    def step(self, action):
        reward = 0.0
        done = False
        info = {}

        curr_time = self.times[self.current_step]
        is_late = curr_time >= time(15, 50)
        
        # 如果选择观望、或是尾盘不允许开仓，直接跳到下一分钟
        if action == 0 or is_late:
            self.current_step += 1
            if self.current_step >= len(self.close_prices) - 1:
                done = True
            return self._get_obs(), 0.0, done, False, info

        # RL 选择开仓 (1 = Long, 2 = Short)
        entry_price = self.close_prices[self.current_step]
        entry_step = self.current_step
        current_atr = self.atr_array[self.current_step]
        
        if action == 1:
            sl_price = entry_price - self.sl_atr * current_atr
            tp_price = entry_price + self.tp_atr * current_atr
            position = 1
        else:
            sl_price = entry_price + self.sl_atr * current_atr
            tp_price = entry_price - self.tp_atr * current_atr
            position = -1

        # 进入内部持仓循环，向前推进时间，直到触发止盈、止损、超时或收盘
        exit_price = np.nan
        while self.current_step < len(self.close_prices) - 1:
            self.current_step += 1
            h = self.high_prices[self.current_step]
            l = self.low_prices[self.current_step]
            c = self.close_prices[self.current_step]
            t = self.times[self.current_step]
            
            # 判断多头退场条件
            if position == 1:
                if l <= sl_price:
                    exit_price = sl_price
                    break
                elif h >= tp_price:
                    exit_price = tp_price
                    break
            # 判断空头退场条件
            elif position == -1:
                if h >= sl_price:
                    exit_price = sl_price
                    break
                elif l <= tp_price:
                    exit_price = tp_price
                    break
                    
            # 检查时间止损 或 尾盘强平
            if (self.current_step - entry_step) >= self.max_holding_bars or t >= time(15, 50):
                exit_price = c
                break

        if np.isnan(exit_price):
            exit_price = self.close_prices[self.current_step]

        # 计算这笔交易的净收益率
        if position == 1:
            trade_return = (exit_price - entry_price) / entry_price
        else:
            trade_return = (entry_price - exit_price) / entry_price
            
        trade_return -= 2 * self.cost  # 扣除一买一卖双边手续费

        # Reward 即为这笔交易的净收益率 (放大以利于神经网络梯度更新)
        reward = trade_return * 100.0

        if self.current_step >= len(self.close_prices) - 1:
            done = True
            
        return self._get_obs(), float(reward), done, False, info


# ---------------------------------------------------------------------------
# 3. 训练与启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    args = parser.parse_args()

    # 1. 准备数据
    days, feature_cols = prepare_rl_data("SPY", "2024-01-01", "2025-12-31")
    
    # 划分训练集和验证集 (前80%作为训练集，后20%验证)
    train_split = int(len(days) * 0.8)
    train_days = days[:train_split]
    eval_days = days[train_split:]

    if args.train:
        print("Initializing RL environments...")
        # 向量化环境，可以加速训练
        train_env = make_vec_env(lambda: PAOptionsDayEnv(train_days, feature_cols), n_envs=4)
        eval_env = make_vec_env(lambda: PAOptionsDayEnv(eval_days, feature_cols), n_envs=1)

        # 评估回调函数，自动保存在验证集上表现最好的模型
        eval_callback = EvalCallback(eval_env, best_model_save_path='./rl_models/',
                                     log_path='./rl_logs/', eval_freq=2000,
                                     deterministic=True, render=False)

        # 核心算法：PPO
        # ent_coef: 探索系数，稍微调大一点防止模型很快陷入“永远空仓”的局部最优
        model = PPO("MlpPolicy", train_env, verbose=1, ent_coef=0.05, learning_rate=3e-4, 
                    n_steps=2048, batch_size=256, tensorboard_log="./rl_tensorboard/")

        print("Starting PPO Training... (Press Ctrl+C to stop manually)")
        try:
            # 训练 300万步
            model.learn(total_timesteps=3000000, callback=eval_callback)
        except KeyboardInterrupt:
            print("Training interrupted manually.")

        model.save("./rl_models/ppo_pa_options_final")
        print("Model saved to ./rl_models/ppo_pa_options_final.zip")
    else:
        print("\nData loaded successfully! Run with '--train' to start PPO training.")
        print("Example: python backtests/train_rl_pa_options.py --train")
