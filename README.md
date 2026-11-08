# GNSS 中断下 USV 转弯过桥洞：入口认证与预测安全滤波

| 文件 | 内容 |
|---|---|
| `GNSS_Bridge_Same_Bridge.ipynb` | 方法与主实验（8 m × 19 m 同桥对比、同航速消融）。方法已冻结。 |
| `01_monte_carlo.ipynb` | Monte Carlo：随机入口状态与海流；错误放行率、误拒率、通过率、误差界覆盖率、认证丢失率 |
| `02_length_sweep.ipynb` | 桥长扫描（8 m 宽）：能过比例、滤波器放行率/撞墙率、ILOS 成功率、EKF 3σ 放行率/撞墙率同图；桥宽 × 桥长认证图 |
| `03_baselines.ipynb` | 对照组：EKF 3σ 裕度安全滤波、纯 INS 推算 + 安全滤波；INS 器件等级扫描 |
| `04_margins_and_tight_bound.ipynb` | 去掉额外余量、收紧误差界后重跑：最坏世界中 EKF 3σ 撞墙，本文方法不撞 |
| `bridge_sim/core.py` | 从主 notebook 原样拷贝的方法代码（不修改） |
| `bridge_sim/fast.py` | 船模导数的等价加速（结果逐位一致） |
| `bridge_sim/ext.py` | 实验扩展：可替换导航器/误差管的闭环、随机场景、事后“能过”标注、EKF 3σ 与纯 INS 误差管 |
| `bridge_sim/metrics.py` | 各比例的定义与 95% Wilson 区间 |
| `tests/test_regression.py` | 检查 `run_case_ext` 默认参数下与原 `run_case` 逐位一致 |
| `results/` | 所有原始数据（每个场景一行）与汇总表，见 `results/README.md` |
| `figures/` | 新实验的图（PNG + PDF） |
| `tools/make_0*.py` | 生成 notebook 的脚本 |

运行：`pip install numpy pandas matplotlib jupyter`，然后依次执行 01 → 02 → 03（03 读取 01 的结果）。
`results/` 中已有 CSV 时 notebook 直接读取；删掉对应 CSV 即重新仿真（4 核约 3 小时）。

所有实验使用同一组误差界参数（`bridge_sim/core.py: Config`），不按案例调整；对比只改方法，世界、种子、初态相同。
