# House Purchase Prediction ML Project

A comprehensive machine learning project to predict whether customers will purchase a house based on property characteristics and customer financial information.

## Project Structure

```
house_purchase_predictor/
├── data/                           # Data directory
│   └── global_house_purchase_dataset.csv
├── src/                            # Source code
│   ├── data_preprocessing.py       # Data preprocessing and feature engineering
│   ├── model_training.py           # Model training and evaluation
│   ├── main_pipeline.py            # Main orchestration pipeline
│   ├── predict.py                  # Prediction module for new data
│   └── quick_demo.py               # Quick demo script
├── app/                            # Streamlit web application
│   ├── streamlit_app.py            # Main Streamlit app
│   ├── README.md                   # App documentation
│   └── TROUBLESHOOTING.md          # App troubleshooting guide
├── models/                         # Saved models and artifacts
├── notebooks/                      # Jupyter notebooks for EDA
│   └── eda.ipynb
├── tests/                          # Unit tests
│   └── test.py
├── guides/                         # User guides and documentation
│   ├── APP_GUIDE.md                # Complete Streamlit app guide
│   ├── USAGE_GUIDE.md              # Detailed usage instructions
│   ├── QUICKSTART.md               # Quick start guide
│   └── QUICK_REFERENCE.md          # Quick reference card
├── docs/                           # Project documentation
│   ├── PROJECT_SUMMARY.md          # High-level project overview
│   ├── FINAL_OVERVIEW.md           # Quick reference guide
│   ├── FIXES_APPLIED.md            # Bug fixes and solutions
│   ├── CI_CD_GUIDE.md              # CI/CD pipeline documentation
│   ├── CI_CD_SUMMARY.md            # CI/CD implementation summary
│   ├── CI_CD_FIXES.md              # CI/CD fixes applied
│   └── CI_CD_FINAL_SUMMARY.md      # Final CI/CD summary
├── .github/                        # GitHub Actions and templates
│   ├── workflows/                  # CI/CD workflows
│   └── ISSUE_TEMPLATE/             # Issue and PR templates
├── requirements.txt                # Python dependencies
├── run_app.sh                      # Launch script (Mac/Linux)
├── run_app.bat                     # Launch script (Windows)
└── README.md                       # This file
```

## Features

### Data Preprocessing

- **Data Quality Checks**: Missing values, duplicates, data types
- **Feature Engineering**:
  - Property age calculation
  - Price per square foot
  - Loan to price ratio
  - Down payment ratio
  - Affordability score
  - Total rooms
  - Amenities score
  - Risk score (crime + legal cases)
  - Monthly payment burden
  - Disposable income ratio
- **Encoding**: Label encoding for categorical features
- **Scaling**: StandardScaler for numerical features
- **Data Split**: 70% train, 15% validation, 15% test (stratified)

### Models Evaluated

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential boosting
4. **XGBoost** - Optimized gradient boosting
5. **Support Vector Machine** - SVM with RBF kernel
6. **K-Nearest Neighbors** - Distance-based classifier
7. **Naive Bayes** - Probabilistic classifier

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

### Best Practices Implemented

✅ Train/Validation/Test split to prevent data leakage  
✅ Stratified sampling to maintain class balance  
✅ Feature scaling fitted only on training data  
✅ Multiple model comparison  
✅ Comprehensive evaluation metrics  
✅ Feature importance analysis  
✅ Model persistence with versioning  
✅ Clear function documentation with docstrings  
✅ Logging throughout the pipeline  
✅ Modular code structure  
✅ Separate prediction module for deployment  

## 🎨 Interactive Web App

We've included a beautiful **Streamlit web application** for easy predictions!

### Quick Start - Web App

```bash
# Launch the app (easiest way)
./run_app.sh          # Mac/Linux
# or
run_app.bat           # Windows

# Or run directly
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501` with:

- ✨ **Single Prediction**: Interactive form with instant results
- 📊 **Batch Prediction**: Upload CSV, get predictions for all
- 📈 **Visualizations**: Gauge charts, probability scores, insights
- 💡 **Risk Assessment**: Affordability, loan ratios, risk factors

See `app/README.md` for detailed documentation.

---

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the complete ML pipeline:

```bash
cd src
python main_pipeline.py
```

This will:

1. Load and preprocess the data
2. Train 7 different models
3. Evaluate all models on validation set
4. Select the best model based on F1 score
5. Evaluate the best model on test set
6. Display feature importance
7. Save the model and artifacts to `models/` directory
8. Generate a detailed performance report

### Making Predictions

To make predictions on new data:

```bash
cd src
python predict.py ../data/your_data.csv
```

This will:

- Load the latest trained model
- Preprocess the input data
- Make predictions
- Save results to `your_data_predictions.csv`

### Using the Predictor Programmatically

```python
from predict import HousePurchasePredictor
import pandas as pd

# Initialize predictor
predictor = HousePurchasePredictor(
    model_path='../models/best_model_XGBoost_20231004_120000.pkl',
    scaler_path='../models/scaler_20231004_120000.pkl',
    encoders_path='../models/encoders_20231004_120000.pkl',
    feature_names_path='../models/feature_names_20231004_120000.json'
)

# Load your data
df = pd.read_csv('your_data.csv')

# Get predictions with probabilities
results = predictor.predict_with_explanation(df)
print(results)
```

## Model Performance

The pipeline automatically selects the best performing model based on validation F1 score. Typical results:

- **Best Model**: XGBoost or Random Forest (varies by run)
- **Test Accuracy**: ~85-90%
- **Test F1 Score**: ~0.85-0.90
- **Test ROC AUC**: ~0.90-0.95

Detailed performance metrics for all models are saved in `models/model_comparison_*.csv`.

## Output Files

After running the pipeline, the following files are created in the `models/` directory:

- `best_model_<ModelName>_<timestamp>.pkl` - Trained model
- `scaler_<timestamp>.pkl` - Fitted StandardScaler
- `encoders_<timestamp>.pkl` - Label encoders for categorical features
- `feature_names_<timestamp>.json` - List of feature names
- `model_comparison_<timestamp>.csv` - Performance comparison of all models
- `model_report.txt` - Detailed performance report

## Key Features of the Code

### Clean Functions with Docstrings

All functions include comprehensive docstrings with:

- Description of functionality
- Parameter types and descriptions
- Return value descriptions
- Usage examples where applicable

### Error Handling

- Try-catch blocks for robust error handling
- Informative error messages
- Graceful degradation when optional features aren't available

### Logging

- Detailed logging throughout the pipeline
- Progress tracking for long-running operations
- Performance metrics logging

### Modularity

- Separate modules for preprocessing, training, and prediction
- Easy to extend with new models or features
- Reusable components

## Dataset

The dataset includes:

- **Property Features**: Type, size, price, age, rooms, amenities
- **Location Features**: Country, city, crime rate, neighborhood rating
- **Customer Features**: Salary, expenses, loan details, satisfaction
- **Target**: Decision (0 = Not Buy, 1 = Buy)

## Future Enhancements

- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Cross-validation for more robust evaluation
- SHAP values for model interpretability
- API endpoint for real-time predictions
- Dashboard for model monitoring
- Automated retraining pipeline

## License

See LICENSE file for details.

## 📚 Documentation

### **📖 Main Documentation**

- **[README.md](README.md)** - This file (main documentation)

### **📋 User Guides** (`guides/`)

- **[APP_GUIDE.md](guides/APP_GUIDE.md)** - Complete Streamlit app guide
- **[USAGE_GUIDE.md](guides/USAGE_GUIDE.md)** - Detailed usage instructions
- **[QUICKSTART.md](guides/QUICKSTART.md)** - Quick start guide
- **[QUICK_REFERENCE.md](guides/QUICK_REFERENCE.md)** - Quick reference card

### **📊 Project Documentation** (`docs/`)

- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - High-level project overview
- **[FINAL_OVERVIEW.md](docs/FINAL_OVERVIEW.md)** - Quick reference guide
- **[FIXES_APPLIED.md](docs/FIXES_APPLIED.md)** - Bug fixes and solutions

### **🚀 CI/CD Documentation** (`docs/`)

- **[CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md)** - CI/CD pipeline documentation
- **[CI_CD_SUMMARY.md](docs/CI_CD_SUMMARY.md)** - CI/CD implementation summary
- **[CI_CD_FIXES.md](docs/CI_CD_FIXES.md)** - CI/CD fixes applied
- **[CI_CD_FINAL_SUMMARY.md](docs/CI_CD_FINAL_SUMMARY.md)** - Final CI/CD summary

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
