import pandas as pd
import numpy as np

r = pd.read_csv('results/20260329v3/ce_zlsma_kama_rule_refined_optimization.csv')
spy = r[r['dataset']=='SPY'].copy()

print(f'SPY total combos: {len(spy)}')
print(f'Best overall Sharpe: {spy["sharpe"].max():.4f}')
print()

# 1) No PA at all
no_pa = spy[
    (spy['pa_or_filter']==False) & (spy['pa_use_mm_target']==False) &
    (spy['pa_regime_filter']==False) & (spy['pa_mag_bar_exit']==False) &
    (spy['pa_use_pa_stops']==False) & (spy['pa_require_signal_bar']==False)
]
b = no_pa.sort_values('sharpe', ascending=False).iloc[0]
print('='*80)
print('1) NO PA RULES (baseline)')
print(f'   Sharpe={b["sharpe"]:.4f} | Return={b["total_return"]*100:.2f}% | '
      f'Win={b["win_rate"]*100:.1f}% | Trades={int(b["trade_count"])} | '
      f'DD={b["max_drawdown"]*100:.2f}% | TS={int(b["time_stop_minutes"])}m | '
      f'TP_ATR={b["tp_atr_multiple"]}')
print(f'   TP%={b["take_profit_rate"]*100:.1f}% SL%={b["stop_loss_rate"]*100:.1f}% '
      f'TS%={b["time_stop_rate"]*100:.1f}%')
print(f'   Avg hold={b["avg_holding_minutes"]:.1f}m')

# 2) Only Signal Bar
only_sb = spy[
    (spy['pa_or_filter']==False) & (spy['pa_use_mm_target']==False) &
    (spy['pa_regime_filter']==False) & (spy['pa_mag_bar_exit']==False) &
    (spy['pa_use_pa_stops']==False) & (spy['pa_require_signal_bar']==True)
]
b = only_sb.sort_values('sharpe', ascending=False).iloc[0]
print()
print('='*80)
print('2) ONLY Signal Bar = True (current best)')
print(f'   Sharpe={b["sharpe"]:.4f} | Return={b["total_return"]*100:.2f}% | '
      f'Win={b["win_rate"]*100:.1f}% | Trades={int(b["trade_count"])} | '
      f'DD={b["max_drawdown"]*100:.2f}% | TS={int(b["time_stop_minutes"])}m | '
      f'TP_ATR={b["tp_atr_multiple"]}')
print(f'   TP%={b["take_profit_rate"]*100:.1f}% SL%={b["stop_loss_rate"]*100:.1f}% '
      f'TS%={b["time_stop_rate"]*100:.1f}%')
print(f'   Avg hold={b["avg_holding_minutes"]:.1f}m')

# 3) Each PA rule ON vs OFF impact
print()
print('='*80)
print('3) EACH PA RULE: best Sharpe with ON vs OFF')
print()

pa_rules = [
    ('pa_or_filter', 'OR Filter'),
    ('pa_use_mm_target', 'MM Target'),
    ('pa_regime_filter', 'Regime Filter'),
    ('pa_mag_bar_exit', 'MAG+EG Exit'),
    ('pa_use_pa_stops', 'PA Stops'),
    ('pa_require_signal_bar', 'Signal Bar'),
]

for col, name in pa_rules:
    on = spy[spy[col]==True].sort_values('sharpe', ascending=False).iloc[0]
    off = spy[spy[col]==False].sort_values('sharpe', ascending=False).iloc[0]
    ds = on['sharpe'] - off['sharpe']
    dr = (on['total_return'] - off['total_return']) * 100
    print(f'  {name:20s}')
    print(f'    ON:  Sharpe={on["sharpe"]:.4f} Ret={on["total_return"]*100:.2f}% '
          f'Win={on["win_rate"]*100:.1f}% Tr={int(on["trade_count"]):4d} '
          f'DD={on["max_drawdown"]*100:.2f}%')
    print(f'    OFF: Sharpe={off["sharpe"]:.4f} Ret={off["total_return"]*100:.2f}% '
          f'Win={off["win_rate"]*100:.1f}% Tr={int(off["trade_count"]):4d} '
          f'DD={off["max_drawdown"]*100:.2f}%')
    print(f'    Delta: Sharpe={ds:+.4f}  Return={dr:+.2f}%')
    print()

# 4) All PA ON
all_pa = spy[
    (spy['pa_or_filter']==True) & (spy['pa_use_mm_target']==True) &
    (spy['pa_regime_filter']==True) & (spy['pa_mag_bar_exit']==True) &
    (spy['pa_use_pa_stops']==True) & (spy['pa_require_signal_bar']==True)
]
if len(all_pa) > 0:
    b = all_pa.sort_values('sharpe', ascending=False).iloc[0]
    print('='*80)
    print('4) ALL PA RULES ON')
    print(f'   Sharpe={b["sharpe"]:.4f} | Return={b["total_return"]*100:.2f}% | '
          f'Win={b["win_rate"]*100:.1f}% | Trades={int(b["trade_count"])} | '
          f'DD={b["max_drawdown"]*100:.2f}% | TS={int(b["time_stop_minutes"])}m | '
          f'TP_ATR={b["tp_atr_multiple"]}')
    print(f'   TP%={b["take_profit_rate"]*100:.1f}% SL%={b["stop_loss_rate"]*100:.1f}% '
          f'TS%={b["time_stop_rate"]*100:.1f}%')
    print(f'   Avg hold={b["avg_holding_minutes"]:.1f}m')

# 5) Top 10
print()
print('='*80)
print('5) TOP 10 SPY BY SHARPE')
print()
top10 = spy.sort_values('sharpe', ascending=False).head(10)
for i, (_, row) in enumerate(top10.iterrows()):
    pa_str = ''
    if row['pa_require_signal_bar']: pa_str += 'SB '
    if row['pa_or_filter']: pa_str += 'OR '
    if row['pa_use_mm_target']: pa_str += 'MM '
    if row['pa_regime_filter']: pa_str += 'RF '
    if row['pa_mag_bar_exit']: pa_str += 'MAG '
    if row['pa_use_pa_stops']: pa_str += 'PS '
    if not pa_str: pa_str = '(none)'
    print(f'  #{i+1:2d} Sharpe={row["sharpe"]:.4f} Ret={row["total_return"]*100:.2f}% '
          f'Win={row["win_rate"]*100:.1f}% Tr={int(row["trade_count"]):4d} '
          f'DD={row["max_drawdown"]*100:.2f}% '
          f'TS={int(row["time_stop_minutes"]):2d}m TP={row["tp_atr_multiple"]:4.2f} '
          f'PA=[{pa_str.strip()}]')

# 6) Average Sharpe by PA config (aggregated)
print()
print('='*80)
print('6) AVERAGE SHARPE BY PA RULE (across all TP/TS combos)')
print()
for col, name in pa_rules:
    avg_on = spy[spy[col]==True]['sharpe'].mean()
    avg_off = spy[spy[col]==False]['sharpe'].mean()
    med_on = spy[spy[col]==True]['sharpe'].median()
    med_off = spy[spy[col]==False]['sharpe'].median()
    print(f'  {name:20s}  ON: avg={avg_on:.4f} med={med_on:.4f}  |  '
          f'OFF: avg={avg_off:.4f} med={med_off:.4f}  |  '
          f'delta_avg={avg_on-avg_off:+.4f}')
