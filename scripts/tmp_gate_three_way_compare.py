import os
import numpy as np
import pandas as pd
import pickle

from scripts.tmp_scan_straddle_wide_pess import load_oos_with_gate, straddle_wide_pess
from core.trainers.constants import L1A_OUTPUT_CACHE_FILE, L1B_OUTPUT_CACHE_FILE
from core.trainers.stack_v2_common import load_output_cache

base = load_oos_with_gate().sort_values(['symbol','time_key']).reset_index(drop=True)

l1a = load_output_cache(L1A_OUTPUT_CACHE_FILE).copy()
l1b = load_output_cache(L1B_OUTPUT_CACHE_FILE).copy()
for x in (l1a, l1b):
    x['time_key'] = pd.to_datetime(x['time_key'])

base = base.merge(l1a[['symbol','time_key','l1a_vol_forecast']], on=['symbol','time_key'], how='left')
base = base.merge(l1b[['symbol','time_key','l1b_edge_pred']], on=['symbol','time_key'], how='left')

trade_rate = float(base['gate_on'].mean())
trade_n = int(base['gate_on'].sum())

score_b = pd.to_numeric(base['l1b_edge_pred'], errors='coerce').fillna(pd.to_numeric(base['l1b_edge_pred'], errors='coerce').median())
score_a = pd.to_numeric(base['l1a_vol_forecast'], errors='coerce').fillna(pd.to_numeric(base['l1a_vol_forecast'], errors='coerce').median())
score_b = score_b.to_numpy(dtype=np.float64)
score_a = score_a.to_numpy(dtype=np.float64)

thr_b = float(np.quantile(score_b, 1.0 - trade_rate))
thr_a = float(np.quantile(score_a, 1.0 - trade_rate))
base['gate_b_only'] = score_b >= thr_b
base['gate_a_only'] = score_a >= thr_a

params = dict(sl_atr=1.0, trail_act=1.0, trail_dist=0.2, max_hold=60)

def run_mask(mask_col):
    parts=[]
    for sym,g in base.groupby('symbol', sort=True):
        g=g.sort_values('time_key').reset_index(drop=True)
        m=g[mask_col].to_numpy(dtype=bool)
        r=straddle_wide_pess(g,m,**params)
        if not r.empty:
            r['symbol']=sym
            parts.append(r)
    if not parts:
        return dict(n=0, ev=np.nan, wr=np.nan, std=np.nan, ambig=np.nan, avg_hold=np.nan)
    allr=pd.concat(parts, ignore_index=True)
    return dict(
        n=int(len(allr)),
        ev=float(allr['total_pnl'].mean()),
        wr=float((allr['total_pnl']>0).mean()),
        std=float(allr['total_pnl'].std()),
        ambig=float(allr['ambig'].mean()),
        avg_hold=float(allr['hold_bars'].mean()),
    )

r_cur=run_mask('gate_on')
r_b=run_mask('gate_b_only')
r_a=run_mask('gate_a_only')

summary=pd.DataFrame([
    {'gate_variant':'current_gate(L2_trade_gate)','mask_rate':float(base['gate_on'].mean()),'mask_n':int(base['gate_on'].sum()),'threshold':np.nan,**r_cur},
    {'gate_variant':'l1b_only(l1b_edge_pred)','mask_rate':float(base['gate_b_only'].mean()),'mask_n':int(base['gate_b_only'].sum()),'threshold':thr_b,**r_b},
    {'gate_variant':'l1a_only(l1a_vol_forecast)','mask_rate':float(base['gate_a_only'].mean()),'mask_n':int(base['gate_a_only'].sum()),'threshold':thr_a,**r_a},
])
summary['ev_cost']=summary['ev']-0.04

overlap=pd.DataFrame([
    {'pair':'current_vs_l1b','jaccard':float((base['gate_on']&base['gate_b_only']).sum()/((base['gate_on']|base['gate_b_only']).sum()+1e-12))},
    {'pair':'current_vs_l1a','jaccard':float((base['gate_on']&base['gate_a_only']).sum()/((base['gate_on']|base['gate_a_only']).sum()+1e-12))},
    {'pair':'l1b_vs_l1a','jaccard':float((base['gate_b_only']&base['gate_a_only']).sum()/((base['gate_b_only']|base['gate_a_only']).sum()+1e-12))},
])

print('=== Gate Generation (Current) ===')
print('current_gate := (1 - l2_decision_neutral) >= trade_threshold_from_l2_meta')
print(f'trade_rate={trade_rate:.6f} trade_n={trade_n}')
print('\n=== Three-way Comparison (matched trade rate, sl=1.0 ta=1.0 td=0.2 max_hold=60) ===')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
print('\n=== Gate Overlap (Jaccard) ===')
print(overlap.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

out_dir=os.path.join(os.getcwd(),'reports'); os.makedirs(out_dir, exist_ok=True)
summary.to_csv(os.path.join(out_dir,'l2_gate_l1a_l1b_three_way_compare.csv'), index=False)
overlap.to_csv(os.path.join(out_dir,'l2_gate_overlap_jaccard.csv'), index=False)
print(f'\nSaved to: {out_dir}')
