# Usage Guide - House Purchase Prediction Model

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Quick Demo (Fast - 30 seconds)

Test the pipeline on a subset of data:

```bash
cd src
python quick_demo.py
```

This runs 3 fast models on 10,000 samples to verify everything works.

### 3. Full Pipeline (Comprehensive - 5-10 minutes)

Train all 7 models on the complete dataset:

```bash
cd src
python main_pipeline.py
```

This will:

- Load and preprocess 200,000 samples
- Train 7 different models
- Evaluate on validation and test sets
- Save the best model
- Generate detailed performance reports

## Code Structure

### `data_preprocessing.py`

Contains all data preprocessing functions:

```python
from data_preprocessing import preprocess_pipeline

# Complete preprocessing in one call
X_train, X_val, X_test, y_train, y_val, y_test, encoders, scaler, feature_names = preprocess_pipeline(
    filepath='../data/global_house_purchase_dataset.csv',
    random_state=42
)
```

**Key Functions:**

- `load_data()` - Load CSV file
- `check_data_quality()` - Analyze data quality
- `engineer_features()` - Create 10 new features
- `encode_categorical_features()` - Label encoding
- `split_data()` - 70/15/15 train/val/test split
- `scale_features()` - StandardScaler normalization
- `preprocess_pipeline()` - Complete pipeline

### `model_training.py`

Contains model training and evaluation:

```python
from model_training import train_and_evaluate_all_models, select_best_model

# Train all models
trained_models, results_df = train_and_evaluate_all_models(
    X_train, y_train, X_val, y_val
)

# Select best model
best_model_name, best_model = select_best_model(results_df, trained_models)
```

**Key Functions:**

- `get_model_configs()` - Get 7 model configurations
- `train_model()` - Train a single model
- `evaluate_model()` - Calculate metrics
- `train_and_evaluate_all_models()` - Train all models
- `select_best_model()` - Select best by F1 score
- `get_feature_importance()` - Analyze feature importance
- `save_model_and_artifacts()` - Save model with metadata
- `generate_detailed_report()` - Create performance report

### `main_pipeline.py`

Orchestrates the complete pipeline:

```python
python main_pipeline.py
```

Runs all 7 steps automatically:

1. Data Preprocessing
2. Model Training (7 models)
3. Model Selection
4. Test Set Evaluation
5. Feature Importance Analysis
6. Save Model & Artifacts
7. Generate Report

### `predict.py`

For making predictions on new data:

```python
from predict import HousePurchasePredictor

# Initialize predictor
predictor = HousePurchasePredictor(
    model_path='../models/best_model_RandomForest_20231004_120000.pkl',
    scaler_path='../models/scaler_20231004_120000.pkl',
    encoders_path='../models/encoders_20231004_120000.pkl',
    feature_names_path='../models/feature_names_20231004_120000.json'
)

# Make predictions
predictions = predictor.predict(new_data_df)
probabilities = predictor.predict_proba(new_data_df)
```

**Command Line Usage:**

```bash
python predict.py ../data/new_data.csv
# Saves predictions to: new_data_predictions.csv
```

## Feature Engineering

The pipeline creates 10 engineered features:

1. **property_age** = 2025 - constructed_year
2. **price_per_sqft** = price / property_size_sqft
3. **loan_to_price_ratio** = loan_amount / price
4. **down_payment_ratio** = down_payment / price
5. **affordability_score** = customer_salary / price
6. **total_rooms** = rooms + bathrooms
7. **amenities_score** = garage + garden
8. **risk_score** = crime_cases_reported + legal_cases_on_property
9. **monthly_payment_burden** = monthly_expenses + (loan_amount / (loan_tenure_years * 12))
10. **disposable_income_ratio** = (customer_salary - monthly_expenses) / customer_salary

## Models Trained

| Model | Description | Typical Performance |
|-------|-------------|-------------------|
| Logistic Regression | Linear baseline | F1: ~0.85 |
| Random Forest | Ensemble of trees | F1: ~0.95 |
| Gradient Boosting | Sequential boosting | F1: ~0.92 |
| XGBoost* | Optimized boosting | F1: ~0.95 |
| SVM | Support vector classifier | F1: ~0.88 |
| K-Nearest Neighbors | Distance-based | F1: ~0.90 |
| Naive Bayes | Probabilistic | F1: ~0.80 |

*XGBoost is optional - pipeline works without it if not installed

## Evaluation Metrics

For each model, we calculate:

- **Accuracy**: Overall correctness
- **Precision**: How many predicted purchases were correct
- **Recall**: How many actual purchases were caught
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve (discrimination ability)
- **Confusion Matrix**: Detailed breakdown of predictions

## Output Files

After running the pipeline, you'll find in `models/`:

```
models/
├── best_model_RandomForest_20231004_120000.pkl      # Trained model
├── scaler_20231004_120000.pkl                       # Feature scaler
├── encoders_20231004_120000.pkl                     # Categorical encoders
├── feature_names_20231004_120000.json               # Feature list
├── model_comparison_20231004_120000.csv             # All model results
└── model_report.txt                                 # Performance report
```

## Example Workflow

### Training

```bash
# 1. Quick test
python quick_demo.py

# 2. Full training
python main_pipeline.py

# Expected output:
# ======================================================================
# HOUSE PURCHASE PREDICTION - MODEL PERFORMANCE REPORT
# ======================================================================
# 
# VALIDATION SET RESULTS (All Models):
# ----------------------------------------------------------------------
#              model_name  accuracy  precision    recall  f1_score  roc_auc
#           Random Forest    0.9580     0.9245    0.9156    0.9200   0.9875
#      Gradient Boosting    0.9523     0.9145    0.9023    0.9084   0.9832
# ...
# 
# BEST MODEL: Random Forest
# F1 Score: 0.9200
```

### Prediction

```bash
# Make predictions on new data
python predict.py ../data/new_customer_data.csv

# Output: new_customer_data_predictions.csv
# Contains: prediction, prediction_label, probability_buy, confidence
```

## Best Practices Implemented

✅ **Data Leakage Prevention**

- Encoders and scalers fit only on training data
- Separate validation and test sets
- Stratified sampling for class balance

✅ **Code Quality**

- Comprehensive docstrings for all functions
- Type hints for parameters
- Modular design with separation of concerns
- Logging throughout

✅ **Model Evaluation**

- Multiple metrics (accuracy, precision, recall, F1, ROC AUC)
- Validation set for model selection
- Separate test set for final evaluation
- Feature importance analysis

✅ **Reproducibility**

- Fixed random seeds (42)
- Versioned model artifacts with timestamps
- Complete preprocessing pipeline saved
- Requirements.txt with package versions

✅ **Production Ready**

- Separate prediction module
- Error handling and logging
- Model persistence
- Easy to deploy predictor class

## Troubleshooting

### XGBoost Import Error

If you see XGBoost errors:

```bash
# On Mac:
brew install libomp

# Or skip XGBoost (pipeline works without it):
# The code automatically handles missing XGBoost
```

### Memory Issues

For large datasets:

```python
# Modify main_pipeline.py to use a subset:
df = pd.read_csv(filepath, nrows=50000)  # Use 50k rows instead of all
```

### Slow Training

Speed up training:

- Use `quick_demo.py` for testing
- Reduce n_estimators in model configs
- Use fewer models (comment out slow ones like SVM)

## Advanced Usage

### Custom Model Configuration

Edit `model_training.py`:

```python
def get_model_configs():
    models = {
        'My Custom Model': {
            'model': YourClassifier(param1=value1),
            'description': 'Your description'
        }
    }
    return models
```

### Custom Features

Edit `data_preprocessing.py`:

```python
def engineer_features(df):
    # Add your custom features
    df['my_new_feature'] = df['col1'] * df['col2']
    return df
```

### Change Data Split

Edit `main_pipeline.py`:

```python
X_train, X_val, X_test, y_train, y_val, y_test, ... = preprocess_pipeline(
    filepath=DATA_PATH,
    test_size=0.20,      # 20% test instead of 15%
    val_size=0.15,       # 15% validation
    random_state=42
)
```

## Performance Tips

1. **Feature Selection**: Remove low-importance features
2. **Hyperparameter Tuning**: Use GridSearchCV on best model
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Cross-Validation**: Use k-fold CV for more robust evaluation
5. **Class Imbalance**: Try SMOTE or class weights if needed

## Contact & Support

For issues or questions:

1. Check the logs in terminal output
2. Review the model_report.txt
3. Check that all dependencies are installed
4. Verify data format matches expected schema
