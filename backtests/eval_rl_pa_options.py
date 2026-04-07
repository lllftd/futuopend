import os
import sys
import warnings
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from datetime import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtests.train_rl_pa_options import prepare_rl_data, PAOptionsDayEnv

warnings.filterwarnings("ignore")

def run_rl_backtest(model_path, symbol="SPY", start_date="2024-01-01", end_date="2025-12-31"):
    print(f"Loading data and features for testing ({start_date} to {end_date})...")
    days, feature_cols = prepare_rl_data(symbol, start_date, end_date)
    
    # Use the last 20% of days for out-of-sample testing
    train_split = int(len(days) * 0.8)
    eval_days = days[train_split:]
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    all_trades = []
    
    print(f"\n--- Running Agent on Test Dataset ({len(eval_days)} days) ---")
    
    # Evaluate on ALL test days
    for test_day_idx, test_day_data in enumerate(eval_days):
        env = PAOptionsDayEnv([test_day_data], feature_cols)
        obs, _ = env.reset()
        
        done = False
        
        while not done:
            prev_step = env.current_step
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            curr_time = test_day_data['time_key'].iloc[prev_step].time()
            is_late = curr_time >= time(15, 50)
            
            if action != 0 and not is_late:
                entry_dt = test_day_data['time_key'].iloc[prev_step]
                entry_price = test_day_data['close'].iloc[prev_step]
                
                exit_dt = test_day_data['time_key'].iloc[env.current_step]
                exit_price = test_day_data['close'].iloc[env.current_step]
                
                direction = "LONG" if action == 1 else "SHORT"
                
                all_trades.append({
                    "entry_time": entry_dt,
                    "entry_price": entry_price,
                    "exit_time": exit_dt,
                    "exit_price": exit_price,
                    "direction": direction,
                    "pnl_pct": reward / 100.0
                })
                
    print(f"\nTotal Trades taken by AI across all {len(eval_days)} test days: {len(all_trades)}")
    
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        wins = df_trades[df_trades["pnl_pct"] > 0]
        losses = df_trades[df_trades["pnl_pct"] < 0]
        
        win_rate = len(wins) / len(df_trades) if len(df_trades) > 0 else 0
        pf = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0
        net_ret = df_trades["pnl_pct"].sum()
        
        print(f"Total Net PnL: {net_ret*100:.3f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Profit Factor: {pf:.2f}")
        
        print("\nFirst 10 Trades:")
        for _, t in df_trades.head(10).iterrows():
            print(f"[{t['direction']}] Entry: {t['entry_time']} @ {t['entry_price']:.2f} | "
                  f"Exit: {t['exit_time'].time()} @ {t['exit_price']:.2f} | "
                  f"Net PnL: {t['pnl_pct']*100:.3f}%")
    else:
        print("The AI decided to take NO TRADES during the entire test period.")

if __name__ == "__main__":
    import glob
    # Find the best model saved by EvalCallback or fallback to final model
    best_models = glob.glob("./rl_models/best_model.zip")
    final_models = glob.glob("./rl_models/ppo_pa_options_final.zip")
    
    model_to_use = best_models[0] if best_models else (final_models[0] if final_models else None)
    
    if model_to_use:
        run_rl_backtest(model_to_use)
    else:
        print("No trained models found in ./rl_models/")