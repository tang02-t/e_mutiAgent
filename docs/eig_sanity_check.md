# EIG 推荐领域一致性抽检（B-2 验收）

> 生成：`python3 scripts/eval/eig_sanity_check.py`，参数 `/Users/ts/Desktop/thu/multi_Agent/data/real/dga/learned_params.json`
> 20 条手工部分观测样例；「预期」为按 DL/T 722 直觉预先写下的可接受征兆集合，命中其一即一致。
> **Top-1 一致率 90%，Top-3 一致率 95%**。预期集合由 AI 预填，需人工复核后在此登记复核人与日期。

| # | 样例 | Top-1 故障 | H(bit) | 动作 | 推荐 Top-3（征兆 / EIG / cost / VoI） | 预期集合 | Top-1 命中 | 理由 |
|---:|---|---|---:|---|---|---|:---:|---|
| 1 | 仅氢气高 | `partial_discharge` | 1.85 | ask | `C2H2_elevated` 0.218/0/0.218<br>`C2H4_elevated` 0.142/0/0.142<br>`gas_rate_rapid` 0.160/1/0.110 | `C2H2_elevated`, `CH4_elevated`, `partial_discharge_alarm` | ✓ | H2 单独升高：局放 vs 受潮 vs 低温过热，关键看 C2H2 / CH4 或局放检测 |
| 2 | 氢乙炔高、无其他气体 | `winding_short_circuit` | 1.20 | ask | `C2H4_elevated` 0.095/0/0.095<br>`TDCG_elevated` 0.054/0/0.054<br>`winding_temp_elevated` 0.088/1/0.038 | `C2H4_elevated`, `CH4_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✓ | 放电已确定，区分匝间短路（伴随过热/温升）与套管/分接开关放电（不伴随） |
| 3 | 乙烯乙烷高、无乙炔 | `overload_overheating` | 0.93 | ask | `CH4_elevated` 0.033/0/0.033<br>`winding_temp_elevated` 0.078/1/0.028<br>`TDCG_elevated` 0.023/0/0.023 | `CH4_elevated`, `CO2_elevated`, `CO_elevated`, `load_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✓ | 过热特征，需确认是否负载/温度导致（过载过热）或固体绝缘参与（CO/CO2） |
| 4 | CO 类信息缺失的过热 | `overload_overheating` | 0.73 | conclude | `winding_temp_elevated` 0.067/1/0.017<br>`load_elevated` 0.051/1/0.001<br>`oil_temp_elevated` 0.035/1/-0.015 | `CO2_elevated`, `CO_elevated`, `load_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✓ | 高温过热已定，区分裸金属过热与固体绝缘过热看 CO/CO2；是否过载看负载与温度 |
| 5 | 全气体正常、用户报油温高 | `insulation_degradation` | 1.59 | conclude | `winding_temp_elevated` 0.064/1/0.014<br>`CO2_elevated` 0.050/1/0.000<br>`load_elevated` 0.044/1/-0.006 | `CO2_elevated`, `CO_elevated`, `gas_rate_rapid`, `load_elevated`, `winding_temp_elevated` | ✓ | 气体正常但油温高：首先排查是否过载或冷却问题，再看产气趋势 |
| 6 | 仅乙炔高 | `partial_discharge` | 1.80 | ask | `C2H4_elevated` 0.154/0/0.154<br>`C2H6_elevated` 0.096/0/0.096<br>`TDCG_elevated` 0.084/0/0.084 | `C2H4_elevated`, `CH4_elevated`, `H2_elevated` | ✓ | 乙炔升高必是放电，H2/C2H4 决定能量等级（三比值法两个比值都缺） |
| 7 | 局放告警、无气体 | `partial_discharge` | 1.83 | ask | `C2H2_elevated` 0.197/0/0.197<br>`C2H4_elevated` 0.149/0/0.149<br>`C2H6_elevated` 0.101/0/0.101 | `C2H2_elevated`, `CH4_elevated`, `H2_elevated`, `TDCG_elevated` | ✓ | 局放告警需油色谱确认（H2 为主、C2H2 低） |
| 8 | 振动异常、无气体 | `overload_overheating` | 1.96 | ask | `C2H2_elevated` 0.235/0/0.235<br>`C2H4_elevated` 0.142/0/0.142<br>`gas_rate_rapid` 0.167/1/0.117 | `C2H2_elevated`, `H2_elevated`, `TDCG_elevated`, `gas_rate_rapid`, `load_elevated` | ✓ | 振动异常对应绕组变形/铁芯松动，先看是否伴随放电气体或负载 |
| 9 | 匝间短路典型气体但无温度 | `winding_short_circuit` | 1.09 | conclude | `load_elevated` 0.010/1/-0.040<br>`CO2_elevated` 0.006/1/-0.044<br>`CO_elevated` 0.002/1/-0.048 | `CO2_elevated`, `CO_elevated`, `load_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✓ | 电弧放电兼过热，需要温度/负载确认匝间短路而非分接开关拉弧 |
| 10 | 低能放电、无 H2 | `partial_discharge` | 1.68 | ask | `winding_temp_elevated` 0.095/1/0.045<br>`H2_elevated` 0.042/0/0.042<br>`oil_temp_elevated` 0.087/1/0.037 | `H2_elevated`, `oil_temp_elevated`, `partial_discharge_alarm`, `winding_temp_elevated` | ✓ | 缺 H2 无法判断 CH4/H2，需补 H2 或局放检测 |
| 11 | 过热+CO 高 | `overload_overheating` | 0.60 | conclude | `winding_temp_elevated` 0.057/1/0.007<br>`load_elevated` 0.042/1/-0.008<br>`oil_temp_elevated` 0.024/1/-0.026 | `CO2_elevated`, `load_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✓ | 固体绝缘参与过热，看 CO2 与温度/负载区分绝缘老化与过载 |
| 12 | 绝缘老化疑似 | `insulation_degradation` | 1.54 | conclude | `winding_temp_elevated` 0.065/1/0.015<br>`oil_temp_elevated` 0.053/1/0.003<br>`load_elevated` 0.045/1/-0.005 | `gas_rate_rapid`, `load_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✓ | CO/CO2 高但烃低：老化 vs 过载，看负载/温度与产气速率 |
| 13 | 已排除乙炔与过热气体 | `partial_discharge` | 1.20 | conclude | `winding_temp_elevated` 0.030/1/-0.020<br>`load_elevated` 0.018/1/-0.032<br>`oil_temp_elevated` 0.015/1/-0.035 | `CO2_elevated`, `CO_elevated`, `gas_rate_rapid`, `partial_discharge_alarm` | ✗ | 只剩 H2 高：局放或受潮，局放检测最有区分度 |
| 14 | 总烃高、气体明细缺失 | `overload_overheating` | 1.88 | ask | `C2H2_elevated` 0.249/0/0.249<br>`C2H4_elevated` 0.132/0/0.132<br>`gas_rate_rapid` 0.177/1/0.127 | `C2H2_elevated`, `C2H4_elevated`, `CH4_elevated`, `H2_elevated` | ✓ | 总烃高需拆分明细，乙炔决定放电 vs 过热 |
| 15 | 负载高、气体缺失 | `overload_overheating` | 1.78 | ask | `C2H2_elevated` 0.229/0/0.229<br>`C2H4_elevated` 0.136/0/0.136<br>`gas_rate_rapid` 0.168/1/0.118 | `C2H4_elevated`, `CO_elevated`, `TDCG_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✗ | 负载高看温度是否跟随、是否已产生过热气体 |
| 16 | 产气快、其他缺失 | `winding_short_circuit` | 1.63 | ask | `C2H2_elevated` 0.200/0/0.200<br>`C2H4_elevated` 0.115/0/0.115<br>`TDCG_elevated` 0.066/0/0.066 | `C2H2_elevated`, `C2H4_elevated`, `H2_elevated`, `TDCG_elevated` | ✓ | 产气速率快说明活跃故障，乙炔决定放电性质 |
| 17 | 套管放电疑似 | `partial_discharge` | 1.30 | ask | `winding_temp_elevated` 0.086/1/0.036<br>`oil_temp_elevated` 0.079/1/0.029<br>`CO2_elevated` 0.058/1/0.008 | `load_elevated`, `oil_temp_elevated`, `partial_discharge_alarm`, `vibration_elevated`, `winding_temp_elevated` | ✓ | 低能放电 + 高 H2：套管 / 局放 / 匝间，看局放检测或温度 |
| 18 | 已排除大多数、只剩 CH4 | `overload_overheating` | 1.48 | ask | `winding_temp_elevated` 0.098/1/0.048<br>`load_elevated` 0.071/1/0.021<br>`oil_temp_elevated` 0.061/1/0.011 | `CO2_elevated`, `CO_elevated`, `load_elevated`, `oil_temp_elevated`, `winding_temp_elevated` | ✓ | CH4 单独高：低温过热，看温度/负载/CO |
| 19 | 匝间短路+已知无过热 | `winding_short_circuit` | 1.25 | conclude | `CO2_elevated` 0.016/1/-0.034<br>`load_elevated` 0.012/1/-0.038<br>`CO_elevated` 0.006/1/-0.044 | `CO2_elevated`, `CO_elevated`, `load_elevated`, `partial_discharge_alarm`, `vibration_elevated` | ✓ | 温度已排除，剩余区分度在负载/振动/局放/CO |
| 20 | 正常气体、无征兆 | `partial_discharge` | 1.42 | conclude | `winding_temp_elevated` 0.047/1/-0.003<br>`oil_temp_elevated` 0.036/1/-0.014<br>`load_elevated` 0.032/1/-0.018 | `CO2_elevated`, `CO_elevated`, `gas_rate_rapid`, `load_elevated`, `oil_temp_elevated`, `partial_discharge_alarm`, `vibration_elevated`, `winding_temp_elevated` | ✓ | 气体全正常时任何现场征兆都是补充信息，任意非气体征兆均可接受 |

## κ 敏感性（无数据征兆的专家 CPT 收缩系数）

| κ | Top-1 一致率 | Top-3 一致率 |
|---:|---:|---:|
| 1.0 | 60% | 70% |
| 0.8 | 70% | 85% |
| 0.6 | 90% | 95% |
| 0.5 | 90% | 95% |
| 0.4 | 90% | 95% |
| 0.3 | 90% | 95% |

κ=1.0 为原专家值；`learn_cpt.py` 默认写入 κ=0.5。κ ≤ 0.6 时一致率稳定，说明结论对具体取值不敏感。

人工复核：<待填写：复核人 / 日期 / 修改的预期集合>