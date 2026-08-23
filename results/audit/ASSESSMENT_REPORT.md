# 膀胱癌 ConvNeXt 项目：补充实验可行性核查

核查日期：2026-08-23

## 结论

现有附件足以确认患者级数据隔离、数据规模和标签分布，并可生成一个保留原独立 test set 的患者级 5 折划分方案。现有附件不足以真实计算独立 test 性能、置信区间、ROC/PR、混淆矩阵或 Grad-CAM，也不足以完成 5 折模型训练和性能汇总。原因是三个数据 CSV 只有标注信息，没有模型预测概率；原图路径不可访问；没有模型权重、模型定义、推理/训练代码或逐折结果。

`tumor_ratio`、`ca_ratio` 和 `label` 均来自标注，不能当作模型输出。用它们计算 AUC 会造成标签泄漏，因此本次没有这样做，也没有从已报告的 validation AUC 反推任何 test 指标。

## 已核实的数据

| 集合 | 患者 | 切片 | source images | patches | 标签0 | 标签1 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 18 | 49 | - | 5,423 | 3,658 | 1,765 |
| Validation | 4 | 8 | - | 996 | 671 | 325 |
| Independent test | 4 | 5 | - | 648 | 442 | 206 |
| 合计 | 26 | 62 | 135 | 7,067 | 4,771 | 2,296 |

- train/validation/test 的患者 ID 没有交叉。
- 以 `uuid + x + y` 作为 patch 键时，没有重复记录，也没有缺失单元格。
- 三个 CSV 的 `image_path` 全部指向 `/home/aistudio/...`，当前环境不存在这些图像，无法进行模型推理或 Grad-CAM。
- 独立 test 只有 4 名患者。即使补齐预测，患者聚类 bootstrap 的 95% CI 也会很不稳定，论文中必须明确报告 cluster 数量、有效 bootstrap 次数和方法；不能把 648 个高度相关 patches 当作 648 个独立患者。
- 当前任务可访问的附件中没有 `PROJECT_SUMMARY.txt`、论文正文、`check_and_report.py`、模型训练代码、权重和原始/patch 图像；这些文件虽出现在 `FILE_MANIFEST.txt` 中，但未实际提供给本任务。

## 六项补充工作的状态

| 项目 | 当前状态 | 说明 |
|---|---|---|
| 1. 真正独立 test 结果 | 无法计算 | 有 test 标签和患者列表，但没有冻结模型输出 `y_score`，也没有可运行权重+图像。 |
| 2. AUC 95% CI | 代码已提供，数值待数据 | `analyze_predictions.py` 对患者进行聚类重采样并计算 patch-level AUC 的 percentile CI；没有预测时不运行。 |
| 3. ROC + PR | 代码已提供，图待数据 | 补齐逐 patch 概率后自动生成 300 dpi 图。 |
| 4. 混淆矩阵及 Sens/Spec/PPV/NPV | 代码已提供，数值待数据 | 阈值必须在 validation 上预先冻结；禁止用 test 标签选阈值。默认 0.5 仅在预注册为最终阈值时使用。 |
| 5. Grad-CAM + TP/TN/FP/FN | 无法生成 | 缺少图像、权重、模型定义和预处理。脚本会先输出候选 patch 清单；最终典型图应由病理医师审阅，而不是只选最“漂亮”的例子。 |
| 6. patient-level repeated split / 5-fold CV | 划分已完成，性能未完成 | 已为 train+validation 的 22 名开发患者生成 5 折清单，并保持 4 名原 test 患者封存。每折仍须从头训练并产生 out-of-fold 预测。 |

## 已生成的可复用文件

- `dataset_audit.json`：机器可读的数据核查结果。
- `patient_summary.csv`：每位患者的 patch 数、阳性/阴性数、切片数和阳性比例。
- `cv_patient_folds.csv`：22 名开发患者的 5 折分配。
- `cv_patch_manifest.csv`：逐 patch 的 5 折清单；原 4 名 test 患者未包含。
- `cv_fold_summary.csv`：每折患者数、patch 数和类别平衡。
- `audit_and_make_patient_folds.py`：重新核查数据并生成患者级折叠。
- `analyze_predictions.py`：补齐逐 patch 预测后生成 test 指标、患者聚类 bootstrap AUC CI、ROC/PR、混淆矩阵及 TP/TN/FP/FN 候选清单。

当前 5 折分配的开发集规模为 22 名患者，各折 4–5 人。该清单是可复现的实验设计，不是交叉验证结果。正式训练时，每一折都必须独立完成训练、模型选择和该折推理；不得复用对该折见过的权重。

## 最少需要补充的文件

### 若只做项目 1–4

最少提供一个 CSV：每个 test patch 一行，包含：

```text
uuid,x,y,y_score
```

其中 `y_score` 必须是冻结后的最终模型对肿瘤类的概率，并覆盖 `bladder_test.csv` 的全部 648 行。还需明确最终分类阈值，以及阈值是在 validation 上如何确定的。若有多个模型/seed，增加 `model_id` 或分别提供文件。

### 若需重新推理并做项目 5

除上述内容外，至少提供：

1. 最终模型 checkpoint（例如 `.pth`/`.pt`）。
2. 完整模型定义及精确预处理：输入尺寸、归一化均值/方差、颜色处理、类别顺序。
3. test CSV 中 `image_path` 所指的 135 个 source images（至少 test 对应的 source images），或已裁好的 test patches，并给出坐标含义和 patch 尺寸。
4. 可运行的推理代码和依赖版本；若模型层名非标准，还需说明 Grad-CAM 目标层。
5. 如要在病例/WSI 层展示，需提供 patch 到 WSI/病例的映射和用于展示的原图。

### 若要完成项目 6 的真实性能

至少再提供：

1. 可运行的训练入口、配置和环境依赖。
2. train+validation 对应图像数据。
3. 每折的 early-stopping/模型选择规则和固定随机种子。
4. 每折独立 checkpoint 或逐 patch out-of-fold `y_score`。
5. 若做 repeated split，预先规定重复次数、train/validation 比例和汇总方法。

## 推荐的执行顺序

先冻结当前 A0 模型和 validation 阈值，对从未参与模型选择的 4 名 test 患者生成一次预测；随后运行 `analyze_predictions.py`。不要在看到 test 结果后再改阈值或选 checkpoint。之后再用 22 名开发患者做 5 折训练，最后仍将原 4 名 test 患者作为一次性独立验证保留。

