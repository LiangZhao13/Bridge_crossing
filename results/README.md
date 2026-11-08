# results/

每个原始 CSV 一行一个场景，列名前缀表示方法：`filter_`（本文）、`ilos07_`（0.7 m/s ILOS）、`ekf3s_`（EKF 3σ 安全滤波）、
`ins_`（纯 INS 2 mg）、`ins_1mg_`、`oracle_`（事后标注）。场景参数（种子、桥长宽、海流、初态、零偏）在前几列，可完全复现。

| 文件 | 来源 | 内容 |
|---|---|---|
| `mc_main_bridge.csv` | 01 | 8×19 m 桥，200 个随机世界：filter / ilos07 / oracle |
| `mc_random_length.csv` | 01 | 8 m 宽、桥长 U[15,50]，240 个随机世界 |
| `mc_summary.csv` | 01 | 各比例与 95% Wilson 区间（间隙阈值 0 与 0.28 m 两种口径） |
| `baselines_main_bridge.csv`, `baselines_random_length.csv` | 03 | 同一批世界上的 ekf3s / ins / ins_1mg |
| `baselines_summary.csv` | 03 | 四种方法的比例对比 |
| `ins_grade_map.csv` | 03 | 主实验世界：INS 加速度零偏等级 × 桥长 的入口认证；以及 box / ekf 参照 |
| `length_sweep.csv` | 02 | 30 个世界 × 10 个桥长（8 m 宽）：filter / ekf3s / ilos07 / oracle |
| `length_sweep_summary.csv` | 02 | 桥长扫描图中每条曲线的比例与区间 |
| `width_length_map.csv`, `width_length_limits.csv` | 02 | 主实验世界：桥宽 × 桥长 的入口认证与事后标注；各宽度最长可认证桥长 |
| `worst_world_oracle.csv` | 02 | 声明界上 4 个最坏世界 × 桥长：入口认证与事后标注 |
| `worst_world_closed_loop.csv` | 02 | 最坏世界中 filter 与 ekf3s 的闭环运行 |

主要列：`*_admitted` 入口是否放行，`*_entry_margin` 入口预测裕度，`*_transit_speed` 所选航速，`*_completed` 是否完成，
`*_collided` 是否撞墙，`*_cert_lost` 认证丢失，`*_min_clearance` 真实最小船体间隙，`*_cov_ratio_{N,E,heading}` 全程 max|真实误差|/误差界，
`*_cov_ratio_E_blind` 仅 GNSS 非良好段，`oracle_best_clearance` 事后最佳航速下的间隙（≤0 为撞墙），`oracle_clear_U` 各航速。
| `variants_gate_map.csv` | 04 | 主实验世界：4 种设置（base / legacy_margins / margins0 / tight）× 桥长 的入口认证 |
| `variants_worst_world.csv`, `variants_worst_world_summary.csv` | 04 | 3 种设置 × 9 个桥长 × 8 个最坏世界：filter / ekf3s / oracle 闭环及汇总 |
| `violation_worst_world.csv`, `violation_worst_world_summary.csv` | 05 | 射流误差 0.22–0.40 m/s × 5 个桥长 × 4 个最坏世界 |
| `violation_random.csv`, `violation_random_summary.csv` | 05 | 01 B 部分前 120 个世界、射流误差放大 2 倍：filter / ekf3s / ilos07 / oracle |
| `sensitivity_gate.csv` | 06 | 4 类误差界 × 扫描值 × 桥长 9–51 m 的入口认证 |
| `sensitivity_check.csv` | 06 | 每个设置在最长认证桥长上的闭环检查（覆盖率、撞墙、认证丢失） |
| `sensitivity_requirements.csv` | 06 | 认证 15/19/25/31 m 所需的各项误差界上限 |
| `a1_check.csv` | 07 | 60 次放行运行的盲段：纵荡/横荡水相对速度误差、艏向误差与其声明界（A1 逐项检查） |
| `compare_boundary_random.csv` | 07 | 120 个随机边界世界（L~U[17,27]）：固定航速 ILOS、只入口检查、入口+调速、完整方法 |
| `compare_worst_world.csv` | 07 | 最坏世界 8 个 × 桥长 15–39 m：同上各策略 |
| `compare_summary.csv` | 07 | 上两组的汇总表 |
| `legacy_2026-11-08_sway0.006/` | — | 横荡速度误差界 0.006 m/s（违反 A1）时的全部结果，见其中 NOTE.md |
| `legacy_2026-11-07/` | — | 旧参数（采样保护 0.05、安全余量 0.15）的首轮结果，见其中 NOTE.md |

设置在 `bridge_sim/ext.py: VARIANTS`；`variant` 列记录每行用的设置（缺省为 base，即统一参数）。
