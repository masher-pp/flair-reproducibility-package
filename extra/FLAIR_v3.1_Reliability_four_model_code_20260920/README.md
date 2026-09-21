# FLAIR v3.1 Reliability 四模型代码副本

来源（只读复制）：

`/Users/yczhao/Desktop/FLAIR-reproducibility-package-v3.1/`

## 包含的模型

- Random Forest（`rf`）
- Extra Trees（`extratrees`）
- Histogram Gradient Boosting（`histgb`）
- XGBoost（`xgb`）

四个模型的统一 HPO、五折分组交叉验证、最终拟合与 Test15 评估逻辑位于：

`src/reliability_model/07_hpo_ml_model_families.py`

其中 RF 的基础训练、特征定义与数据读取逻辑位于：

`src/reliability_model/03_train_rf_leaf1_sqrt_oob.py`

同时保留了 v3.1 中另外三份直接实例化 RF 的消融与 HPO 代码：

- `04_ablate_redundant_features.py`
- `05_hpo_rf_random_csv.py`
- `05_hpo_rf_staircase.py`

其余 Python 文件是上述代码直接或间接导入的本地依赖。`src/main_model/` 被保留，是因为特征生成依赖主模型接口。

## 范围说明

本文件夹仅收集代码和环境依赖说明，没有复制原始数据、训练结果、模型权重或缓存。因此它是代码副本，不是可立即独立运行的完整数据包。如需执行，应在原始 FLAIR v3.1 目录结构及相应数据/模型文件就绪的环境中运行。

原始 v3.1 文件未被修改。
