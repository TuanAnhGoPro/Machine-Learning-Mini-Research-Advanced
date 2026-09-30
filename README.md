# SIT307 Task 11.1HD – Reproducing and Improving DASMcC on PTB-XL

**Student:** Tuan Anh Dao
**Unit:** SIT307 Machine Learning, Deakin University
**Paper reproduced:** N. Sinha et al., "DASMcC: Data Augmented SMOTE Multi-Class Classifier for Prediction of Cardiovascular Diseases Using Time Series Features," *IEEE Access*, vol. 11, 2023.

**Video presentation:** [YouTube link]

This project has two parts. In Part 1, I reproduce the DASMcC paper (16 ECG time-series features, SMOTE, five classifiers) under two protocols: the one the paper actually used (SMOTE before the train/test split) and the one it describes (SMOTE only inside the training folds). In Part 2, I propose **LF-MBF** (Leakage-Free, Multi-lead Median-Beat Features), which uses patient-wise cross-validation and 103 clinically meaningful features taken from all 12 ECG leads.

---

## 1. Folder structure

```
SIT307_11_1HD/
├── SIT307_11_1HDfinal.ipynb   # main notebook (all code, already run, outputs visible)
├── README.md                  # this file
├── requirements.txt           # Python packages
├── ptbxl_database.csv         # PTB-XL metadata (labels, patient IDs, folds)
├── scp_statements.csv         # PTB-XL diagnostic code table
├── records100/                # PTB-XL 100 Hz ECG signals (.dat + .hea files)
├── cache/                     # saved features, so re-runs skip feature extraction
├── figures/                   # all plots used in the report (created by the notebook)
└── results/                   # all result tables as CSV (created by the notebook)
```

The PTB-XL files sit in the same folder as the notebook. This is the default setting (`DATA_DIR = Path(".")` in the Configuration cell).

## 2. Installation

I used Python 3.12 with Anaconda on Windows, but any Python 3.10+ should work.

```bash
# (optional) create a clean environment
conda create -n sit307 python=3.12 -y
conda activate sit307

# install the packages
pip install -r requirements.txt
```

The notebook also has an install cell at the top (`%pip install ...`). If you run it, restart the kernel afterwards.

## 3. Dataset

The project uses **PTB-XL v1.0.3** from PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/

The notebook only needs three items: `ptbxl_database.csv`, `scp_statements.csv` and the `records100/` folder. I did not include `records500/` (the 500 Hz version), because the paper and my code only use the 100 Hz signals.

If `records100/` is not in this repository because of GitHub's size limits, you can get it in either of these ways:

- **My copy:** [OneDrive link to records100]
- **Official source:** download the ZIP from the PhysioNet link above, then copy `ptbxl_database.csv`, `scp_statements.csv` and `records100/` into the same folder as the notebook.

PTB-XL is released under the CC BY 4.0 licence (P. Wagner et al., *Scientific Data*, 2020).

## 4. How to run and reproduce the results

1. Open `SIT307_11_1HDfinal.ipynb` in Jupyter Notebook or JupyterLab.
2. Check that the Configuration cell prints `PTB-XL found: True`.
3. Choose **Kernel → Restart & Run All**.

All random steps use `SEED = 42`, so the results should match the report.

**Running time.** A full run from scratch took about 6–7 hours on my laptop. Most of that (about 5.5 hours) is scikit-learn's `GradientBoostingClassifier` in the 10-fold CV of Experiment 1B, because it only uses one CPU core. Some ways to save time:

- **Keep the `cache/` folder.** It already holds the extracted features, so the notebook skips feature extraction. Delete it if you want to re-extract everything from the raw signals.
- **Quick test:** set `QUICK_RUN = True` in the Configuration cell to run the whole pipeline on a 1,500-record subset in a few minutes. The numbers will then differ from the report.
- **Skip slow models in Exp 1B:** remove `"GradientBoost"` (and `"CatBoost"`) from `MODELS_1B`.
- **Keep the machine awake.** If the computer sleeps or the browser disconnects during a long run, the outputs may not appear in the notebook. You can also run it from a terminal, which always saves the outputs into the file:
```bash
  jupyter nbconvert --to notebook --execute --inplace SIT307_11_1HDfinal.ipynb --ExecutePreprocessor.timeout=-1
```
- If parallel feature extraction fails on Windows, set `N_JOBS = 1`.

## 5. Notebook sections

| Section | What it does |
|---|---|
| 0 | Setup: imports, configuration, helper functions |
| 1 | Recomputes metrics from the paper's own confusion matrices (Fig. 7 vs Table 12) |
| 2 | Loads PTB-XL, keeps single-label records, class distribution |
| 3 | Extracts the paper's 16 features with NeuroKit2 (assumptions A1–A8) |
| 4 | **Exp 1A**: the paper's actual protocol (SMOTE before split) + leakage check |
| 5 | **Exp 1B**: the paper's described protocol (10-fold CV, SMOTE inside folds) |
| 6 | Part 1 summary |
| 7 | **Part 2**: LF-MBF features and patient-wise 10-fold CV |
| 8 | Wilcoxon test, per-class results, feature importance, final comparison |

## 6. Expected key results

After running, `results/final_summary.csv` should contain these numbers (XGBoost):

| Setting | Accuracy | F1 | AUC |
|---|---|---|---|
| Paper, Table 12 | 93.0 | 93.0 | – |
| Exp 1A (SMOTE before split) | 86.3 | 86.3 | 97.6 |
| Exp 1B (SMOTE inside folds) | 69.4 | 50.3 | 81.5 |
| Part 2 – Baseline (patient-wise) | 69.0 | 49.9 | 81.3 |
| **Part 2 – Proposed LF-MBF** | **80.7** | **71.3** | **93.6** |

The proposed method beats the baseline on all 10 folds (one-sided Wilcoxon p = 0.001).

## 7. Output files

**`figures/`**: `class_distribution.png`, `peak_detection_demo.png`, `intervals_demo.png`, `exp1a_vs_paper.png`, `exp1a_confusion_matrices.png`, `exp1a_roc_xgb.png`, `exp1a_leakage.png`, `exp1b_confusion_matrices.png`, `exp1b_roc_xgb.png`, `part1_summary_f1.png`, `median_beats_by_class.png`, `part2_boxplot_f1.png`, `part2_confusion_matrices.png`, `part2_roc_proposed.png`, `part2_per_class_f1.png`, `part2_feature_importance.png`

**`results/`**: `paper_check.csv`, `class_distribution.csv`, `exp1a_overall.csv`, `exp1a_vs_paper.csv`, `exp1a_leakage.csv`, `exp1b_mean.csv`, `exp1b_std.csv`, `exp1b_per_class_XGBoost.csv`, `part1_summary.csv`, `part2_mean.csv`, `part2_std.csv`, `part2_wilcoxon.csv`, `part2_per_class.csv`, `part2_top20_features.csv`, `final_summary.csv`, plus per-class tables for every model in Exp 1A.

## 8. GenAI acknowledgement

I used Claude (Anthropic) to help plan the notebook structure, draft and debug parts of the code, and explain some concepts. I ran every experiment myself, checked all the numbers against my own outputs, and I understand and can explain every part of the code.
