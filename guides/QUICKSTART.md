# 🚀 Quick Start Guide

## ✅ Path Issue Fixed

The scripts now use absolute paths and work from anywhere in your project.

---

## 🎯 Three Ways to Run

### 1. Quick Demo (30 seconds) ✨ RECOMMENDED FOR TESTING

```bash
# From project root or anywhere
python src/quick_demo.py

# Or with venv
.venv/bin/python src/quick_demo.py
```

**Output:**

```
✅ Demo completed successfully!
   Random Forest: F1 1.0000
   Logistic Regression: F1 0.8620
   Naive Bayes: F1 0.6510
```

---

### 2. Full Pipeline (5-10 minutes) - Train All Models

```bash
# From project root
python src/main_pipeline.py

# Or with venv
.venv/bin/python src/main_pipeline.py
```

**This will:**

- ✅ Load 200,000 samples
- ✅ Train 6 models (7 if XGBoost available)
- ✅ Save best model to `models/` directory
- ✅ Generate performance report

---

### 3. Make Predictions

```bash
# From project root
python src/predict.py data/your_data.csv
```

**Output:** `your_data_predictions.csv`

---

## 📊 Expected Results

### Quick Demo (10k samples)

```
MODEL COMPARISON (Validation Set)
         model_name  accuracy  precision  recall  f1_score  roc_auc
      Random Forest  1.000000   1.000000    1.00  1.000000 1.000000
Logistic Regression  0.934667   0.874286    0.85  0.861972 0.984910
        Naive Bayes  0.742667   0.482574    1.00  0.650995 0.990538

BEST MODEL: Random Forest (F1: 1.0000)
```

### Full Pipeline (200k samples)

```
VALIDATION SET RESULTS (All Models):
            model_name  accuracy  precision   recall  f1_score  roc_auc
         Random Forest  1.000000   1.000000 1.000000  1.000000 1.000000
     Gradient Boosting  1.000000   1.000000 1.000000  1.000000 1.000000
Support Vector Machine  0.986567   0.980789 0.960492  0.970534 0.998771
   Logistic Regression  0.942235   0.866799 0.885239  0.875922 0.985918

BEST MODEL: Random Forest or Gradient Boosting
TEST SET: 100% Accuracy on 30,000 samples
```

---

## 🔧 Troubleshooting

### ❌ "FileNotFoundError: No such file or directory"

**Fixed!** ✅ Scripts now use absolute paths.

### ❌ "XGBoost not available"

**Optional!** The pipeline works with 6 models. XGBoost is a bonus.

To install XGBoost on Mac:

```bash
brew install libomp
pip install xgboost
```

### ❌ "Module not found"

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Where Are the Outputs?

After running the full pipeline, check:

```
models/
├── best_model_Random_Forest_*.pkl       ← Trained model
├── scaler_*.pkl                         ← Feature scaler
├── encoders_*.pkl                       ← Categorical encoders
├── feature_names_*.json                 ← Feature names
├── model_comparison_*.csv               ← All model results
└── model_report.txt                     ← Performance report
```

---

## 🎯 What to Do Next

### ✅ Step 1: Test the Installation

```bash
python src/quick_demo.py
```

Should complete in ~30 seconds with no errors.

### ✅ Step 2: Train on Full Dataset (Optional)

```bash
python src/main_pipeline.py
```

Takes 5-10 minutes. Trains all 6 models.

### ✅ Step 3: Make Predictions

```bash
# Use the trained model
python src/predict.py data/your_new_data.csv
```

### ✅ Step 4: Run Tests

```bash
python tests/test.py
```

All 6 tests should pass.

---

## 💡 Pro Tips

### Run from Any Directory

```bash
# Works from anywhere!
python /path/to/ml_project_template/src/quick_demo.py

# Or change directory first
cd ~/personal/project/ml_project_template
python src/quick_demo.py
```

### Use Virtual Environment

```bash
# Create venv
python -m venv .venv

# Activate
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run scripts
python src/quick_demo.py
```

### Check Model Performance

```bash
# View detailed report
cat models/model_report.txt

# View all model results
open models/model_comparison_*.csv
```

---

## 📞 Common Commands

```bash
# Quick test (30 sec)
python src/quick_demo.py

# Full training (5-10 min)
python src/main_pipeline.py

# Make predictions
python src/predict.py data/your_data.csv

# Run tests
python tests/test.py

# View report
cat models/model_report.txt

# Count lines of code
wc -l src/*.py tests/*.py
```

---

## ✅ Success Checklist

After running `quick_demo.py`, you should see:

- ✅ "Loading subset of data (10,000 rows)..."
- ✅ "DATA PREPROCESSING"
- ✅ "TRAINING MODELS (Fast models only for demo)"
- ✅ "Training Logistic Regression..."
- ✅ "Training Random Forest..."
- ✅ "Training Naive Bayes..."
- ✅ "MODEL COMPARISON (Validation Set)"
- ✅ "BEST MODEL: Random Forest"
- ✅ "✅ Demo completed successfully!"

---

## 🎉 You're All Set

Your ML pipeline is ready to use. The path issues are fixed and everything works from any directory.

**Next Steps:**

1. ✅ Run `python src/quick_demo.py` to verify
2. ✅ Run `python src/main_pipeline.py` for full training (optional)
3. ✅ Use `python src/predict.py your_data.csv` to make predictions

**Questions?** Check the other docs:

- `README.md` - Complete documentation
- `USAGE_GUIDE.md` - Detailed usage examples
- `PROJECT_SUMMARY.md` - High-level overview
- `FINAL_OVERVIEW.md` - Results and achievements

---

**Status: ✅ READY TO USE**
