import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbbuild import build
cells = [
('md', r"""
# 07 回应审稿式意见：A1 逐项核验，以及“合理固定航速”对照

**1. A1 逐项核验。** 位置误差盒覆盖只说明合成后的盒子够大。这里逐项核对船模辅助导航的对水纵向/横向速度误差和艏向误差，
时间窗为入口判断时刻到运行结束，每 0.02 s 计算一次“误差 / 当时有效的界”，取最大值（不是两个峰值相除）。
核验历史：横向界 0.006 m/s 时断星段实测 0.0119（超界）；改为 0.015 后断星段不再超界，但入口到断星之间、出断星以后
（GNSS 可用、船在转向/加速/重新捕获）实测最大 0.0208。因此 A1 按阶段声明：**GNSS 可用时 0.025 m/s，断星时 0.015 m/s**
（`Config.speed_error_lateral_gnss / speed_error_lateral`，误差盒按距上次 GNSS 融合的时间切换），所有实验已按此重跑。

**2. 和“已经选得合理的航速”比，方法在哪里做出更好的决策？** 同一批接近准入边界的场景，比较：

| 策略 | 入口判断 | 航速 | 桥下 |
|---|---|---|---|
| 固定航速 ILOS（1.4 / 1.5 / 1.6 / 1.8 m/s） | 无，总是进 | 固定 | ILOS |
| 固定 1.5 + 入口筛选（`gate_fix15`） | 只在 1.5 m/s 认证，不过则后撤 | 1.5 | ILOS |
| 固定 1.5 + 入口筛选 + 在线滤波（`gate_fix15f`） | 同上 | 1.5 | 安全滤波 |
| 固定 U + 入口筛选（由逐航速入口裕度复算，U = 1.4/1.5/1.6/1.8） | 只在 U 认证（裕度 ≥ 0.02 m） | U | ILOS |
| 入口筛选 + 选速（`gate_speed`） | 14 个候选航速认证 | 选裕度最大者 | ILOS |
| 完整方法（`filter`） | 同上 | 同上 | 安全滤波 |

固定航速 ILOS 的结果直接取自事后标注（同一世界、同一进近，入口切到该航速、全程 ILOS）。“不可通过”指 14 个 ILOS 候选航速都没有通过。
注意：入口筛选用备份策略做预测，放行后 `gate_*` 实际执行 ILOS，所以“只做入口筛选”在这里只作消融，不是独立的安全方案。

场景：随机世界 120 个，桥长 U[17, 27] m（8 m 宽，主方法放行边界附近）；以及声明界上的最坏世界 8 个 × 9 个桥长（15–39 m）。
"""),
('code', r"""
import sys, time
from pathlib import Path
from dataclasses import replace
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from bridge_sim.ext import sample_scenario, run_batch, pmap, a1_check, DEFAULT_SCENARIO
from bridge_sim.metrics import wilson

RES = ROOT/'results'; RES.mkdir(exist_ok=True)
FIG = ROOT/'figures'; FIG.mkdir(exist_ok=True)
WORKERS = 4
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

def worst_worlds8(L):
    return [replace(DEFAULT_SCENARIO, length=float(L), width=8.0, jet_amplitude_error=0.22*sj,
                    current_north_bias=0.010*so, current_east_bias=0.011*so, tide_amplitude=0.004,
                    gyro_bias=0.00055*so, wall_shift=-0.08*sj, initial_cross_track=0.5*so, initial_heading_error=h)
            for sj in (1, -1) for so in (1, -1) for h in (0.1, -0.1)]

def boundary_random(n=120):
    rng = np.random.default_rng(2029)
    out = []
    for i in range(n):
        L = float(rng.uniform(17.0, 27.0))
        out.append(sample_scenario(rng, 40000 + i, length=round(L, 2), width=8.0))
    return out

def cached(path, scen, methods):
    if path.exists():
        print('loading', path.name); return pd.read_csv(path)
    t0 = time.time()
    df = pd.DataFrame(run_batch(scen, methods, workers=WORKERS, progress_every=40, log_path=str(RES/'progress.log')))
    df.to_csv(path, index=False); print(f'{path.name}: {len(df)} runs in {time.time()-t0:.0f} s')
    return df
"""),
('md', "## 1. A1 逐项核验（断星段）"),
('code', r"""
path = RES/'a1_check.csv'
if path.exists():
    a1 = pd.read_csv(path); print('loading', path.name)
else:
    rng = np.random.default_rng(2027)
    mcB = []
    for i in range(240):
        L = float(rng.uniform(15.0, 50.0)); mcB.append(sample_scenario(rng, 20000 + i, length=round(L, 2), width=8.0))
    adm = pd.read_csv(RES/'mc_random_length.csv')['filter_admitted'].fillna(False).astype(bool).values
    scen = [s for s, a in zip(mcB, adm) if a] + [s for L in (15, 18, 21) for s in worst_worlds8(L)] + [DEFAULT_SCENARIO]
    a1 = pd.DataFrame(pmap(a1_check, [dict(s=s) for s in scen], workers=WORKERS))
    a1.to_csv(path, index=False)
n_all = len(a1)
a1 = a1[a1.blind.astype(bool)]
rows = []
for name, e, r, b in (('surge speed (m/s)', 'surge_err', None, 'surge_bound'), ('sway speed (m/s)', 'sway_err', 'sway_ratio', None),
                      ('heading (rad)', None, 'heading_ratio', None)):
    for ph in ('blind', 'gnss'):
        ratio = a1[f'{e}_{ph}']/a1[b] if r is None else a1[f'{r}_{ph}']
        rows.append({'term': name, 'phase': 'GNSS blind' if ph == 'blind' else 'GNSS available (after gate)',
                     'max error': a1[f'{e}_{ph}'].max() if e else np.nan,
                     'declared bound': (a1[b].min() if b else (a1[f'sway_bound_{ph}'].iloc[0] if r == 'sway_ratio' else np.nan)),
                     'max ratio (per time)': ratio.max(), 'runs over bound': int((ratio > 1).sum())})
summary = pd.DataFrame(rows).set_index(['term', 'phase'])
print(f'{n_all} closed-loop runs (admitted MC-B worlds + 24 worst worlds + main world), {len(a1)} reach the GNSS-blind segment')
summary.round(5)
"""),
('md', "## 2. 固定航速 vs 入口筛选 vs 选速 vs 完整方法"),
('code', r"""
METHODS = ('filter', 'gate_speed', 'gate_fix15', 'gate_fix15f', 'oracle')
rb = cached(RES/'compare_boundary_random.csv', boundary_random(), METHODS)
ww = cached(RES/'compare_worst_world.csv', [s for L in range(15, 40, 3) for s in worst_worlds8(L)], METHODS)
for d in (rb, ww):
    assert 'error' not in d or d['error'].isna().all()
FIXED = ['1.4', '1.5', '1.6', '1.8']
from bridge_sim.core import Config
ADM = Config().admission_margin

def table(df):
    N = len(df); feas = df['oracle_best_clearance'] > 0
    rows = []
    for U in FIXED:
        clr = df[f'oracle_clear_{U}']; coll = clr <= 0
        rows.append(dict(strategy=f'fixed {U} m/s ILOS', admitted=N, passed=int((~coll).sum()), collided=int(coll.sum()),
                         admitted_collided=int(coll.sum()), cert_lost=0, rejected_feasible=0))
    for U in FIXED:
        g = df[f'filter_gate_margin_{U}'] >= ADM; ok = df[f'oracle_clear_{U}'] > 0
        rows.append(dict(strategy=f'fixed {U} + gate (recomputed)', admitted=int(g.sum()), passed=int((g & ok).sum()),
                         collided=int((g & ~ok).sum()), admitted_collided=int((g & ~ok).sum()), cert_lost=0,
                         rejected_feasible=int((~g & feas).sum())))
    for m, name in (('gate_fix15', 'fixed 1.5 + gate'), ('gate_fix15f', 'fixed 1.5 + gate + filter'),
                    ('gate_speed', 'gate + speed selection'), ('filter', 'full method')):
        adm = df[m + '_admitted'].fillna(False).astype(bool); col = df[m + '_collided'].astype(bool)
        done = df[m + '_completed'].astype(bool) & ~col
        rows.append(dict(strategy=name, admitted=int(adm.sum()), passed=int(done.sum()), collided=int(col.sum()),
                         admitted_collided=int((adm & col).sum()), cert_lost=int(df[m + '_cert_lost'].sum()),
                         rejected_feasible=int((~adm & feas).sum())))
    t = pd.DataFrame(rows).set_index('strategy')
    t['N'] = N; t['feasible'] = int(feas.sum()); t['infeasible'] = int((~feas).sum())
    t['collision upper 95%'] = [wilson(k, N)[1] for k in t['collided']]
    return t

tab_rb, tab_ww = table(rb), table(ww)
pd.concat({'random, L~U[17,27]': tab_rb, 'worst worlds, 15-39 m': tab_ww}).to_csv(RES/'compare_summary.csv')
print('Random worlds near the admission boundary'); display(tab_rb)
print('Worst worlds'); display(tab_ww)
"""),
('code', r"""
# Where do the decisions differ?  per length in the worst worlds
def by_len(df):
    out = {}
    for U in ('1.5', '1.8'):
        out[f'fixed {U} collided'] = (df[f'oracle_clear_{U}'] <= 0).groupby(df.length).sum()
    out['feasible'] = (df.oracle_best_clearance > 0).groupby(df.length).sum()
    for m in ('gate_fix15', 'gate_speed', 'filter'):
        out[f'{m} admitted'] = df[m + '_admitted'].fillna(False).astype(bool).groupby(df.length).sum()
        out[f'{m} collided'] = df[m + '_collided'].astype(bool).groupby(df.length).sum()
    out['filter speed'] = df.groupby('length')['filter_transit_speed'].agg(lambda x: ','.join(sorted(set(f'{v:.1f}' for v in x))))
    return pd.DataFrame(out)
by_len(ww)
"""),
('code', r"""
# Cases where online filtering changes the outcome at the same admitted speed
for name, df in (('random', rb), ('worst', ww)):
    a = df['gate_speed_admitted'].fillna(False).astype(bool)
    diff = a & (df['gate_speed_collided'].astype(bool) != df['filter_collided'].astype(bool))
    print(name, 'gate+speed vs full method, same admission, different collision outcome:', int(diff.sum()))
    if diff.any():
        display(df.loc[diff, ['length', 'jet_amplitude_error', 'filter_transit_speed', 'gate_speed_min_clearance',
                              'gate_speed_collided', 'filter_min_clearance', 'filter_collided']].round(3))
    clr = df.loc[a & ~df['gate_speed_collided'].astype(bool) & ~df['filter_collided'].astype(bool),
                 ['gate_speed_min_clearance', 'filter_min_clearance']]
    print('   mean clearance gain from the online filter when both pass:',
          round(float((clr.filter_min_clearance - clr.gate_speed_min_clearance).mean()), 3), 'm over', len(clr), 'runs')
"""),
('code', r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout='constrained')
for ax, (name, t) in zip(axes, (('Random worlds, L~U[17,27] m', tab_rb), ('Worst worlds, 15-39 m', tab_ww))):
    x = np.arange(len(t))
    ax.bar(x - .2, t['passed']/t['N'], .4, color='#40886c', label='passed')
    ax.bar(x + .2, t['collided']/t['N'], .4, color='#a83b34', label='collided')
    ax.set_xticks(x, t.index, rotation=35, ha='right', fontsize=8)
    ax.set(title=name, ylabel='fraction of runs', ylim=(0, 1.05)); ax.grid(alpha=.2, axis='y')
axes[0].legend(fontsize=8)
fig.savefig(FIG/'compare_strategies.png', dpi=200); fig.savefig(FIG/'compare_strategies.pdf')
plt.show()
"""),
('md', r"""
## 结论

**A1 逐项检查**（59 个算例，其中 51 个进入断星段；入口判断时刻到运行结束，逐时刻“误差 / 当时的界”取最大）：
- 断星段：纵荡 0.79、横荡 0.92（0.0141 / 0.015）、艏向 0.81。
- GNSS 可用段（入口后）：纵荡 0.73、横荡 0.83（0.0208 / 0.025）、艏向 0.88。
- 没有任何一项在任何时刻超界。分段声明后 A1 在入口到出桥全程成立。

**与固定航速对比**（“不可通过”= 14 个 ILOS 候选航速都没通过）：
- 随机边界世界（L~U[17,27]，120 个，全部能过）：固定 1.4/1.5/1.6 m/s ILOS 全部通过，1.8 m/s 撞 5 次。
  加入口筛选后，固定 1.4/1.5/1.6/1.8 分别放行 23/30/37/29 次；入口筛选 + 选速放行 39 次。全部 0 撞墙。
  选速比固定 1.5 多放行 9 次，比最好的固定航速（1.6）只多 2 次。
- 最坏世界（误差取在声明界上，15–39 m，72 次，8 次不可通过）：固定 1.5 m/s 撞 8 次，恰好是 36、39 m 的 8 个不可通过世界；
  1.4 m/s 撞 12 次、1.6 m/s 撞 13 次（各有 4、5 次在能过的世界），1.8 m/s 撞 36 次。所有带入口筛选的策略都只放行 15、18 m（16 次），0 撞墙。
- 这一组里只有桥长在变，所以事后设“桥长 ≤ 33 m 才进”的门槛也能做到 0 撞墙、通过 56 次。误差盒能否比桥长门槛更会区分，见 08。
- 代价是误拒：最坏世界 21 m 及以上的 48 个能过的世界全部被拒；随机世界只放行 39/120。

**各部分的贡献**：
- 入口筛选：在误差取到界上时拒绝了全部不可通过的世界；没有它，固定航速选错时还会在能过的世界撞墙。注意它用备份策略做预测、放行后执行 ILOS，这里只作消融。
- 航速选择：在相同准入规则下挽回一小部分通行机会（随机世界 +9 次对固定 1.5，+2 次对固定 1.6；最坏世界无增益）。
- 在线滤波：没有改变任何一次撞墙结果。同航速配对中平均间隙 +0.51 m（随机 39 次，33 增 6 减）、+0.12 m（最坏世界 16 次）；
  固定 1.5 m/s 后再加滤波只有 +0.03 m（随机）和 −0.06 m（最坏世界）。论文中定位为间隙改善的次要贡献。

"""),
]
build(str(pathlib.Path(__file__).parents[1]/'07_fixed_speed_and_A1.ipynb'), cells)
