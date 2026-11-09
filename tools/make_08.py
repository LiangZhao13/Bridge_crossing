import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbbuild import build
cells = [
('md', r"""
# 08 同桥长、同桥宽：误差盒是否比“桥长门槛”更会区分该放与该拒

07 的最坏世界组里只有桥长在变，所以事后设一个“桥长 ≤ 33 m 才进”的门槛也能做到 0 碰撞。
这里固定桥的几何（8 m 宽，桥长 18 / 24 / 30 m 各一组），只改变其余条件，看误差盒的入口判断能否区分同一座桥上该放行和该拒绝的世界。

两组世界（`bridge_sim/ext.py: sample_same_geometry`）：

| 组 | 船已知、每个世界不同的量 | 船不知道的量 |
|---|---|---|
| A：已知海流与入口状态 | 海图背景流（东向 0–0.20 m/s）、海图射流幅值 0–0.50 m/s、入口横偏 ±1 m、入口艏向 ±0.15 rad；声明射流误差界固定 0.22 | 射流误差 U[−0.22, 0.22] 及其余界内误差 |
| B：另加各桥位的海图质量 | 同 A，另外每个世界有自己的声明射流误差界 B ~ U[0.05, 0.35]（代表不同桥位的海流图质量，方法知道） | 射流误差 ±B·U[0.6, 1.0]（在该世界自己的界内） |

比较的决策规则（同一世界、同一进近）：

- **桥长门槛**：只看桥长，门槛内总是以固定 1.5 m/s ILOS 进入（07 中最好的固定航速）。在一个固定桥长上，它要么全放、要么全拒。
- **本文方法**：误差盒入口认证 + 选速 + 在线滤波。
- **EKF 3σ**：同一滤波器，误差管换成 3σ。

“能过”= 14 个 ILOS 候选航速中至少一个通过（事后参照）；“不可通过”= 14 个都没通过。
"""),
('code', r"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from bridge_sim.ext import sample_same_geometry, run_batch

RES = ROOT/'results'; RES.mkdir(exist_ok=True)
FIG = ROOT/'figures'; FIG.mkdir(exist_ok=True)
WORKERS = 4
LENGTHS = (18.0, 24.0, 30.0)
METHODS = ('filter', 'ekf3s', 'oracle')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

def worlds(group, n, seed0):
    rng = np.random.default_rng(2030 if group == 'A' else 2031)
    out = []
    for L in LENGTHS:
        for i in range(n):
            out.append(sample_same_geometry(rng, seed0 + len(out), L, 8.0,
                                            declared_jet=None if group == 'A' else (0.05, 0.35)))
    return out

def cached(path, scen):
    if path.exists():
        print('loading', path.name); return pd.read_csv(path)
    t0 = time.time()
    df = pd.DataFrame(run_batch(scen, METHODS, workers=WORKERS, progress_every=60, log_path=str(RES/'progress.log')))
    df.to_csv(path, index=False); print(f'{path.name}: {len(df)} runs in {time.time()-t0:.0f} s')
    return df

sA, sB = worlds('A', 40, 60000), worlds('B', 60, 61000)
A = cached(RES/'same_geometry_A.csv', sA)
B = cached(RES/'same_geometry_B.csv', sB)
for d, sc in ((A, sA), (B, sB)):
    assert 'error' not in d or d['error'].isna().all()
    d['declared_jet'] = [dict(s.cfg_overrides).get('jet_amplitude_error_bound', 0.22) for s in sc]
"""),
('md', "## 1. 每个桥长上的决策与结果"),
('code', r"""
def auc(score, label):
    # probability that a random feasible world scores higher than a random infeasible one
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.concat([pos, neg]).rank()
    return float((r.iloc[:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

def per_length(df):
    rows = []
    for L, g in df.groupby('length'):
        feas = g.oracle_best_clearance > 0
        fix_ok = g['oracle_clear_1.5'] > 0
        r = dict(length=L, N=len(g), feasible=int(feas.sum()), infeasible=int((~feas).sum()),
                 **{'length rule (1.5 m/s, admit all): passed': int(fix_ok.sum()),
                    'length rule: collided': int((~fix_ok).sum())})
        for m, name in (('filter', 'proposed'), ('ekf3s', 'EKF 3σ')):
            adm = g[m + '_admitted'].fillna(False).astype(bool); col = g[m + '_collided'].astype(bool)
            done = g[m + '_completed'].astype(bool) & ~col
            r.update({f'{name}: admitted': int(adm.sum()), f'{name}: passed': int(done.sum()),
                      f'{name}: collided': int(col.sum()),
                      f'{name}: admitted infeasible': int((adm & ~feas).sum()),
                      f'{name}: refused feasible': int((~adm & feas).sum()),
                      f'{name}: AUC margin→feasible': auc(g[m + '_entry_margin'], feas)})
        r['Spearman(proposed margin, best true clearance)'] = g['filter_entry_margin'].rank().corr(g.oracle_best_clearance.rank())
        rows.append(r)
    return pd.DataFrame(rows).set_index('length').T

tA, tB = per_length(A), per_length(B)
pd.concat({'A: known currents & entry state': tA, 'B: + site-specific declared jet bound': tB}, axis=1).to_csv(RES/'same_geometry_summary.csv')
print('Group A'); display(tA.round(3))
print('Group B'); display(tB.round(3))
"""),
('md', r"""
## 2. 哪些已知量决定了放行？哪些未知量决定了结果？

对每个桥长分别计算入口裕度、真实最佳间隙与各变量的 Spearman 相关。
"""),
('code', r"""
KNOWN = ['jet_amplitude', 'mapped_current_east', 'initial_cross_track', 'initial_heading_error', 'declared_jet']
UNKNOWN = ['jet_amplitude_error', 'current_east_bias', 'wall_shift']
def drivers(df):
    out = {}
    for L, g in df.groupby('length'):
        g = g.assign(abs_cross_track=g.initial_cross_track.abs(), abs_heading=g.initial_heading_error.abs(),
                     abs_jet_error=g.jet_amplitude_error.abs())
        cols = ['jet_amplitude', 'mapped_current_east', 'abs_cross_track', 'abs_heading', 'declared_jet',
                'jet_amplitude_error', 'abs_jet_error']
        out[(L, 'entry margin')] = g[cols].rank().corrwith(g.filter_entry_margin.rank())
        out[(L, 'true best clearance')] = g[cols].rank().corrwith(g.oracle_best_clearance.rank())
    return pd.DataFrame(out).round(2)
print('Group A'); display(drivers(A))
print('Group B'); display(drivers(B))
"""),
('code', r"""
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained', sharey='row')
for row, (name, df) in enumerate((('A', A), ('B', B))):
    for ax, (L, g) in zip(axes[row], df.groupby('length')):
        adm = g.filter_admitted.fillna(False).astype(bool)
        x = g.declared_jet if name == 'B' else g.jet_amplitude
        ax.scatter(x[adm], g.oracle_best_clearance[adm], c='#40886c', s=18, label='proposed: admitted')
        ax.scatter(x[~adm], g.oracle_best_clearance[~adm], c='#a83b34', s=18, marker='x', label='proposed: refused')
        fail15 = g['oracle_clear_1.5'] <= 0
        ax.scatter(x[fail15], g.oracle_best_clearance[fail15], facecolors='none', edgecolors='k', s=60,
                   label='fixed 1.5 m/s collides')
        ax.axhline(0, color='k', lw=.8)
        ax.set(title=f'Group {name}, L = {L:.0f} m, width 8 m',
               xlabel='declared jet-error bound (m/s)' if name == 'B' else 'mapped jet amplitude (m/s)',
               ylabel='best true clearance, any of 14 speeds (m)')
        ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=7)
fig.savefig(FIG/'same_geometry.png', dpi=200); fig.savefig(FIG/'same_geometry.pdf')
plt.show()
"""),
('md', r"""
## 结论

数字取自上面的实际运行（统一参数；横荡速度误差界 GNSS 可用时 0.025 m/s、断星时 0.015 m/s；8 m 宽）。

**A 组：只改变已知海流和入口状态（结果不好）**
- 在 18/24/30 m 上，120 个世界全部能过；固定 1.5 m/s 的桥长门槛全部放行也 0 撞墙、通过 120 次。
- 本文方法 18 m 放行 40/40，24 m 和 30 m 全部拒绝（通过 40 次）；EKF 3σ 全部放行、0 撞墙。
- 入口裕度主要由**已知**射流幅值决定（Spearman −0.94～−0.98），而真实间隙几乎不受它影响（−0.08～−0.32），真正决定结果的是**未知**射流误差的大小（−0.81～−0.97）。入口裕度与真实最佳间隙的相关只有 0.05～0.24。
- 解读：在声明误差界固定时，误差盒在同一座桥上基本退化为桥长门槛，而且会因为海图射流强而多拒（误差盒里的射流梯度耦合项按已知射流幅值放大）。已知的入口偏差和背景流在本仿真里几乎不影响能否通过，所以 A 组没有“该拒”的世界可供区分。这一组不支持“误差盒比桥长门槛更会区分”。

**B 组：各桥位海流图精度不同（方法知道各自的声明射流误差界 B）**

| 桥长 | 18 m | 24 m | 30 m |
|---|---|---|---|
| 不可通过 / 世界数 | 0/60 | 0/60 | 5/60 |
| 桥长门槛（固定 1.5 m/s 全放）撞墙 | 0 | 1 | 7 |
| 本文方法 放行 / 撞墙 | 41 / 0 | 25 / 0 | 12 / 0 |
| EKF 3σ 放行 / 撞墙 | 60 / 0 | 52 / 0 | 39 / 0 |

- 只看桥长的规则要做到 0 撞墙，只能在 18 m 放行（通过 60 次）；本文方法在三个桥长上 0 撞墙、通过 78 次，在 24、30 m 仍能放行海流图精度高的桥。
- 入口裕度与真实最佳间隙的 Spearman 相关 0.80～0.91；30 m 上以入口裕度区分能过/不可通过的 AUC 为 0.92。放行与否主要由声明界 B 决定（−0.97～−0.98）。
- EKF 3σ 在 B 组同样 0 撞墙且放行更多（151 次），因为这里的真实误差都在声明界内、并且最多用到 1.0 倍。它的弱点在误差取到界上或超界时（04、05）。

**结论**：误差盒相对桥长门槛的价值，在于把**每座桥已知的不确定度**（海流图质量、传感器等级）换算成放行决策；当不确定度固定、只改变已知海流和入口状态时，它没有比桥长门槛更好，反而更保守。论文应如实写这一点，并把“海流图质量不同的桥位”作为误差盒的主要适用场景。

"""),
]
build(str(pathlib.Path(__file__).parents[1]/'08_same_geometry.ipynb'), cells)
