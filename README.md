# GNSS 中断下 USV 转弯过桥洞：入口认证与预测安全滤波

| 文件 | 内容 |
|---|---|
| `RESEARCH_SUMMARY.md` | 研究总结：问题、方法、假设与命题、全部实验结果与局限 |
| `GNSS_Bridge_Same_Bridge.ipynb` | 方法与主实验（8 m × 19 m 同桥对比、同航速消融）。方法已冻结。 |
| `01_monte_carlo.ipynb` | Monte Carlo：随机入口状态与海流；错误放行率、误拒率、通过率、误差界覆盖率、认证丢失率 |
| `02_length_sweep.ipynb` | 桥长扫描（8 m 宽）：能过比例、滤波器放行率/撞墙率、ILOS 成功率、EKF 3σ 放行率/撞墙率同图；桥宽 × 桥长认证图 |
| `03_baselines.ipynb` | 对照组：EKF 3σ 裕度安全滤波、纯 INS 推算 + 安全滤波；INS 器件等级扫描 |
| `04_margins_and_tight_bound.ipynb` | 余量消融与误差界收紧（以统一参数为基准）；最坏世界中本文方法与 EKF 3σ 的闭环 |
| `05_assumption_violated.ipynb` | 假设被违反：真实射流误差超出声明界 |
| `06_sensor_sensitivity.ipynb` | 传感器/海流图等级敏感性：各声明误差界 → 最长可认证桥长 |
| `07_fixed_speed_and_A1.ipynb` | A1 逐项检查（入口到出桥逐时刻）；与固定航速 ILOS、固定航速 + 入口筛选、入口筛选 + 选速的对比 |
| `08_same_geometry.ipynb` | 同桥长、同桥宽：误差盒入口判断 vs 只看桥长的门槛 vs EKF 3σ |
| `bridge_sim/core.py` | 从主 notebook 原样拷贝的方法代码（不修改） |
| `bridge_sim/fast.py` | 船模导数的等价加速（结果逐位一致） |
| `bridge_sim/ext.py` | 实验扩展：可替换导航器/误差管的闭环、随机场景、事后“能过”标注、EKF 3σ 与纯 INS 误差管 |
| `bridge_sim/metrics.py` | 各比例的定义与 95% Wilson 区间 |
| `tests/test_regression.py` | 检查 `run_case_ext` 默认参数下与原 `run_case` 逐位一致 |
| `results/` | 所有原始数据（每个场景一行）与汇总表，见 `results/README.md` |
| `figures/` | 新实验的图（PNG + PDF） |
| `tools/make_0*.py` | 生成 notebook 的脚本（结论中的数字随结果手动更新） |

运行：`pip install numpy pandas matplotlib jupyter`，然后依次执行 01 → 08（03、05、07 读取 01 的结果）。
`results/` 中已有 CSV 时 notebook 直接读取；删掉对应 CSV 即重新仿真（4 核约 3 小时）。

**统一参数**（2026-11-08 起）：海图配准 0.08 m（真实误差项）、采样保护 0.02 m（船体点横向速度上限 0.83 m/s × 0.02 s）、无额外安全余量；入口裕度阈值 0.02 m（最佳航速预测裕度 ≥ 0.02 m 才放行）。横荡水相对速度误差界按阶段声明：GNSS 可用时 0.025 m/s、断星时 0.015 m/s（07 实测最大分别为 0.0208 和 0.0141）。

所有实验使用同一组误差界参数（`bridge_sim/core.py: Config`），不按案例调整；对比只改方法，世界、种子、初态相同。
