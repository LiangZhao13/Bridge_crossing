"""Rate definitions shared by the notebooks.

feasible (hindsight label, "能过"): plain ILOS, same world and same approach, switched
at the gate to SOME candidate speed passes with true hull clearance > thr; a run that
the evaluated method itself completes without contact also counts as feasible.
"""
import math
import numpy as np
import pandas as pd

# clearance the method itself budgets for (chart registration 0.08 + sample guard 0.02)
METHOD_BUDGET = 0.10


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k/n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (max(0., centre - half), min(1., centre + half))


def _rate(k, n):
    lo, hi = wilson(int(k), int(n))
    return dict(k=int(k), n=int(n), rate=(k/n if n else np.nan), ci_lo=lo, ci_hi=hi)


def feasible(df, m=None, thr=0.0):
    f = df['oracle_best_clearance'] > thr
    if m is not None and m + '_completed' in df:
        ok = df[m + '_completed'] & ~df[m + '_collided']
        f = f | (ok & (df[m + '_min_clearance'] > thr))
    return f


def covered(df, m):
    return (df[m + '_cov_ratio_N'] <= 1) & (df[m + '_cov_ratio_E'] <= 1) & (df[m + '_cov_ratio_heading'] <= 1)


def method_rates(df, m, thr=0.0):
    """All rates for one gated method m (column prefix)."""
    N = len(df)
    adm = df[m + '_admitted'].fillna(False).astype(bool)
    col = df[m + '_collided'].astype(bool)
    done = df[m + '_completed'].astype(bool) & ~col
    lost = df[m + '_cert_lost'].astype(bool)
    feas = feasible(df, m, thr)
    out = {
        'accept (放行率)': _rate(adm.sum(), N),
        'pass (通过率)': _rate(done.sum(), N),
        'collision (撞墙率)': _rate(col.sum(), N),
        'false accept: admitted & collided (错误放行, 按放行数)': _rate((adm & col).sum(), adm.sum()),
        'false accept: admitted & infeasible (错误放行, 按放行数)': _rate((adm & ~feas).sum(), adm.sum()),
        'false reject (误拒, 按能过数)': _rate((~adm & feas).sum(), feas.sum()),
        'bound coverage (误差界覆盖率, 全程)': _rate(covered(df, m).sum(), N),
        'certification loss (认证丢失, 按放行数)': _rate((adm & lost).sum(), adm.sum()),
        'feasible (能过比例)': _rate(feas.sum(), N),
    }
    return out


def rates_table(df, methods, thr=0.0):
    rows = []
    for m in methods:
        for name, r in method_rates(df, m, thr).items():
            rows.append(dict(method=m, metric=name, threshold_m=thr, **r))
    return pd.DataFrame(rows)


def ilos_rates(df, m='ilos07', thr=0.0):
    N = len(df)
    col = df[m + '_collided'].astype(bool)
    done = df[m + '_completed'].astype(bool) & ~col
    rows = [dict(method=m, metric='pass (通过率)', threshold_m=thr, **_rate(done.sum(), N)),
            dict(method=m, metric='collision (撞墙率)', threshold_m=thr, **_rate(col.sum(), N)),
            dict(method=m, metric='bound coverage (误差界覆盖率, 全程)', threshold_m=thr,
                 **_rate(covered(df, m).sum(), N))]
    return pd.DataFrame(rows)


def fmt_table(t):
    t = t.copy()
    t['rate'] = (100*t['rate']).map('{:.1f}%'.format)
    t['95% CI'] = [f"[{100*a:.1f}, {100*b:.1f}]%" for a, b in zip(t.pop('ci_lo'), t.pop('ci_hi'))]
    t['k/n'] = t.pop('k').astype(str) + '/' + t.pop('n').astype(str)
    return t
