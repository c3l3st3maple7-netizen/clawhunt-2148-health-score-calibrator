# 【S6-CAL-003】健康分校准虚构数据生成器 — 交付说明

## 一、交付内容

| 交付物 | 位置 |
|---|---|
| 生成脚本 | `calib/generate_synthetic_health.py`（以 `python -m calib.generate_synthetic_health` 运行） |
| 默认配置 YAML | `configs/default.yaml` |
| `ground_truth.json` 样例 | `deliver/run1/ground_truth.json` |
| 单元/集成测试 | `tests/`（pytest，68 项全部通过） |
| 说明文档 | `README.md`、`docs/schema.md`（两表逐列口径） |
| 验收证据 | `deliver/`（两份运行结果、SHA-256 清单、验收报告、AC2 统计截图、测试日志） |
| 参考校准实现 | `examples/recover_weights.py`（演示如何从生成数据中找回真实权重） |

运行环境：Python 3.11+，**仅标准库**，无网络请求，无第三方 SDK。
本次验证环境：Python 3.12.16 / Linux x86_64 / pytest 9.2.0（仅测试用）。
默认参数（300 家企业 × 3 个会所 × 24 个月）单次生成约 0.6 秒。

## 二、运行方式

```bash
# 生成数据（无需安装任何依赖）
python -m calib.generate_synthetic_health --config configs/default.yaml
# 输出：runs/default/{health_snapshot.csv, churn_label.csv, ground_truth.json, summary.json}

# 指定输出目录 / 覆盖参数
python -m calib.generate_synthetic_health --config configs/default.yaml \
    --out runs/run1 --seed 7 --companies 500 --months 36

# 测试
pip install -r requirements.txt
python -m pytest            # 68 passed

# 一键重建全部验收证据
./scripts/deliver.sh
```

所有参数均来自 YAML（企业数、会所数、月数、年流失率、随机种子、属性抽样比例、
子项噪声、缺失窗口等）。未知键、越界值、非法权重映射会直接报错并给出行号，
不会静默使用默认值。

## 三、验收对照

| 验收标准 | 结果 | 证据 |
|---|---|---|
| 1. 同一种子运行两次，输出文件 SHA-256 相同 | **通过** — 4 个输出文件逐一相同 | `deliver/acceptance.log`、`deliver/sha256.txt` |
| 2. 默认参数下年流失率在 10%–14% 之间 | **通过** — 36/300 = **12.00%**；模型期望 12.0000% | `deliver/ac2-statistics.png`（统计输出截图）、`summary.json` |
| 3. 指定会所空间使用数据连续缺失 3 个月 | **通过** — `SYN-SITE-03`，2025-01～2025-03，300 行为 NULL，100 家企业每家恰好 3 个连续月 | `tests/test_generation.py`、`deliver/acceptance.log` |

原始输出（完整日志见 `deliver/acceptance.log`）：

```
AC1  same seed -> identical SHA-256 for every output file
     health_snapshot.csv    identical  1f3084ae30a4a2fd364dd030fa77682e5da54717bc07bb66f7ea6c7ee063eb09
     churn_label.csv        identical  35b661a72191edaa83019ae58062a058ec5e9aed35d3dbf1a48676e7c707327e
     ground_truth.json      identical  b7005da2dc54bccff2ab320f3f9502b6e7b80a48d89cf4870fc72e6110f1c012
     summary.json           identical  c1b432b6dca5982016f56a6deffc7f9e147d9f1b26e9392a34ce64fe1ea17072
     => PASS

AC2  annual churn rate under the default configuration
     churners 36 / companies 300 = 12.00%   window 2025-01..2025-12 (12 months)
     model expected annual rate 12.0000%
     monthly probability mean 1.2217% (min 0.0022%, max 41.7352%)
     acceptance window 10%-14% -> PASS

AC3  one club's space utilisation missing for three consecutive months
     club SYN-SITE-03 (index 2)
     months 2025-01, 2025-02, 2025-03
     flagged rows 300; companies 100; per-company runs [3]
     space_raw and health_score are NULL on all flagged rows -> PASS
```

## 四、关键实现说明

1. **潜在活跃度与季节因子**：每家企业一个 AR(1) 潜在变量（φ=0.85），1 月、2 月
   叠加 −0.70 的节假日下降，五个子项原始值由「基数 + 载荷 × 活跃度 + 企业级噪声
   + 月度噪声」生成并截断在 0，因此淡季在全部子项上都可见（默认数据中
   `event_raw` 1–2 月均值 10.1，其他月份 12.1）。
2. **标准化只有一处实现**：按子项计算非缺失行总体 z 分数，截断 ±3，再映射为
   `50 + 15·z`（0–100）。`health_score` 为五项得分的加权平均；缺项时为 NULL。
   同一函数可直接被生产评分复用。
3. **截距 b0 用二分法求解**：目标是「模拟概率的**实际**年流失率」
   `mean_i(1 − Π_m(1 − p))` 等于配置值，而不是 `1−(1−月度)^12` 的近似；
   求解后 `mean` 与目标误差小于 1e-9。
4. **真实权重**：event 0.55、interaction 0.42、resource 0.30、opportunity 0.48、
   space 0.22，与 b0、标准化参数、缺项时的重归一化权重一并写入
   `ground_truth.json`；另含每家企业年流失概率、标签与最终潜在活跃度，便于校准
   方法验证能否找回。
5. **缺失处理**：指定会所连续 3 个月 `space_raw`/`space_score`/`health_score`
   为 NULL，`data_flags` 置 bit 0；该期间 logit 使用其余四项权重按原总和重归一化，
   避免缺项本身把流失概率压低。
6. **标签抽样**：默认 `bernoulli`，每家企业按自身年流失概率独立抽取，月份按月度
   风险抽取（符合任务「流失概率 = logistic(…)」的定义）。可选 `exact_count`
   将流失家数固定为 `round(目标 × 企业数)`。
7. **确定性**：单一 `random.Random(seed)`，固定抽样顺序；输出不含时间戳、路径或
   主机名，UUID 由种子与行键 `uuid5` 生成；浮点定点输出、`\n` 换行，因此换目录
   运行也逐字节一致。
8. **两表口径**：按项目统一约定包含 `id/org_id/site_id/created_at/created_by/
   updated_at/deleted_at`，金额无关；企业仅有 `SYN-0001` 形式代号，不含任何名称、
   联系方式或客户名称。若贵方 DDL 列名不同，只需改 `calib/schema.py` 两处元组。

## 五、测试与校准示例

- `python -m pytest`：**68 passed in 27.89 s**（覆盖 YAML 子集、配置校验、列口径、
  三项验收、缺失规则、季节性、评分算术、标识一致性、CLI 退出码、无 site-packages
  运行、禁用 socket 运行、权重找回）。
- `examples/recover_weights.py`（离散时间风险模型，300 家企业 / 36 个流失事件）：
  截距 −5.48（真值 −5.645），权重 MAE 0.197，与真值相关系数 0.81。样本量较小时
  个别权重误差较大（本样本中 resource、space 被低估）；如需更紧的估计，可用
  `--companies 3000 --months 36` 生成更大规模数据。

## 六、备注

- 代码仓库（public，匿名可读，无需邀请）：
  https://github.com/c3l3st3maple7-netizen/clawhunt-2148-health-score-calibrator
  分支 `main`，交付提交 `9c5200a2d06859d739f0e0f28d1153a08d8dc763`。该提交包含全部交付物
  （`calib/`、`configs/`、`tests/`、`docs/`、`examples/`、`scripts/` 与 `deliver/` 下全部验收证据）。
  验证命令：`git clone <上述地址> && cd clawhunt-2148-health-score-calibrator && python -m pytest`
  （Python 3.11+，生成器本身只用标准库；pytest 仅测试用）。
- 交付物中不含客户名称与任何真实企业、个人信息；本单信息仅用于完成本单。
- 如需调整会所数量、缺失窗口位置或增加多个缺失块，修改 `configs/default.yaml`
  即可，无需改代码。
