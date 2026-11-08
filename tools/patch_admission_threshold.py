"""One-off: apply admission_margin = 0.02 m to existing result CSVs.

The admission rule only changes runs whose best gate margin lies in [0, 0.02):
the same best speed is chosen whenever that margin is >= 0.02, so every other
row is unchanged. Closed-loop rows in that band are re-simulated; gate-only rows
are relabelled (admitted = False, speed = NaN). Run from the repo root."""
import ast, glob, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from dataclasses import fields
from bridge_sim.ext import Scenario, evaluate, summarize

THR = 0.02
SCEN_FIELDS = [f.name for f in fields(Scenario)]


def scenario_from_row(r):
    kw = {k: r[k] for k in SCEN_FIELDS if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))}
    kw['seed'] = int(kw['seed'])
    if 'cfg_overrides' in kw and isinstance(kw['cfg_overrides'], str):
        kw['cfg_overrides'] = ast.literal_eval(kw['cfg_overrides'])
    return Scenario(**kw)


for f in sorted(glob.glob('results/*.csv')):
    d = pd.read_csv(f)
    changed = 0
    for c in [c for c in d.columns if c.endswith('entry_margin')]:
        pre = c[:-len('entry_margin')]
        band = (d[c] >= 0) & (d[c] < THR)
        if not band.any():
            continue
        closed_loop = (pre + 'collided') in d.columns
        for i in d.index[band]:
            if closed_loop:
                m = pre.rstrip('_')
                row = evaluate(scenario_from_row(d.loc[i]), (m,))
                for k, v in row.items():
                    if k.startswith(pre):
                        if k not in d.columns:
                            d[k] = np.nan
                        d.at[i, k] = v
            else:
                d.at[i, pre + 'admitted'] = False
                if pre + 'speed' in d.columns:
                    d.at[i, pre + 'speed'] = np.nan
            changed += 1
            print(f, pre or '(gate)', i, 'margin', round(float(d.at[i, c]), 4), 'now admitted', d.at[i, pre + 'admitted'])
    if changed:
        d.to_csv(f, index=False)
