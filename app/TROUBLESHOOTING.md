# 🔧 Streamlit App - Troubleshooting Guide

## ✅ **Issue Fixed: Model Loading Error**

### **Problem:**

```
Error loading model: [Errno 2] No such file or directory: 
'models/best_model_Random_Forest_151009.pkl'
```

### **Root Cause:**

The timestamp extraction was incorrect. Model files have format:

```
best_model_Random_Forest_20251004_151009.pkl
```

But the code was only extracting `151009` instead of `20251004_151009`.

### **Solution:**

✅ **FIXED!** The code now correctly extracts the full timestamp.

The app now:

1. Finds the latest model file
2. Extracts the complete timestamp (date + time)
3. Locates matching scaler, encoders, and feature files
4. Loads all artifacts correctly

---

## 🚀 **How to Run the App**

### **Quick Start:**

```bash
# From project root
./run_app.sh          # Mac/Linux
# or
run_app.bat           # Windows
```

### **Or directly:**

```bash
streamlit run app/streamlit_app.py
```

### **Expected Output:**

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 🐛 **Common Issues & Solutions**

### **1. Model Not Found**

**Error:**

```
No trained model found. Please run main_pipeline.py first.
```

**Solution:**

```bash
# Train the model first
cd src
python main_pipeline.py

# This creates files in models/ directory:
# - best_model_Random_Forest_YYYYMMDD_HHMMSS.pkl
# - scaler_YYYYMMDD_HHMMSS.pkl
# - encoders_YYYYMMDD_HHMMSS.pkl
# - feature_names_YYYYMMDD_HHMMSS.json
```

### **2. Streamlit Not Installed**

**Error:**

```
streamlit: command not found
```

**Solution:**

```bash
pip install streamlit
# or
pip install -r requirements.txt
```

### **3. Import Errors**

**Error:**

```
ModuleNotFoundError: No module named 'plotly'
```

**Solution:**

```bash
# Install all dependencies
pip install -r requirements.txt
```

### **4. Port Already in Use**

**Error:**

```
OSError: [Errno 48] Address already in use
```

**Solution:**

```bash
# Use a different port
streamlit run app/streamlit_app.py --server.port 8502

# Or kill the existing process
lsof -ti:8501 | xargs kill -9  # Mac/Linux
```

### **5. Model Files Mismatch**

**Error:**

```
Scaler file not found: scaler_20251004_151009.pkl
```

**Cause:**
Model and preprocessing files have different timestamps.

**Solution:**

```bash
# Delete old model files and retrain
rm models/*.pkl models/*.json models/*.csv

# Retrain
python src/main_pipeline.py
```

### **6. Prediction Errors**

**Error:**

```
Error making prediction: ...
```

**Solutions:**

- Check all input fields are filled
- Verify numerical inputs are valid
- Ensure CSV format matches template (for batch)
- Check model was trained on same feature set

---

## ✅ **Verification Steps**

### **1. Check Model Files:**

```bash
ls -la models/

# Should see:
# best_model_Random_Forest_YYYYMMDD_HHMMSS.pkl
# scaler_YYYYMMDD_HHMMSS.pkl
# encoders_YYYYMMDD_HHMMSS.pkl
# feature_names_YYYYMMDD_HHMMSS.json
```

### **2. Test Model Loading:**

```python
python -c "
from pathlib import Path
import sys
sys.path.append('src')
from predict import HousePurchasePredictor

models_dir = Path('models')
model_files = list(models_dir.glob('best_model_*.pkl'))
latest = max(model_files, key=lambda x: x.stat().st_mtime)
parts = latest.stem.split('_')
timestamp = '_'.join(parts[-2:])

print(f'Found model: {latest.name}')
print(f'Timestamp: {timestamp}')
print('All files exist:', all([
    (models_dir / f'scaler_{timestamp}.pkl').exists(),
    (models_dir / f'encoders_{timestamp}.pkl').exists(),
    (models_dir / f'feature_names_{timestamp}.json').exists()
]))
"
```

### **3. Test Streamlit:**

```bash
streamlit --version
# Should output: Streamlit, version 1.28.0 or higher
```

---

## 📊 **Understanding Model Files**

### **File Naming Convention:**

```
Pattern: [type]_[timestamp].pkl
Example: best_model_Random_Forest_20251004_151009.pkl

Breakdown:
- best_model_Random_Forest = Model type
- 20251004 = Date (YYYYMMDD)
- 151009 = Time (HHMMSS)
```

### **Required Files (all with same timestamp):**

1. **Model**: `best_model_Random_Forest_YYYYMMDD_HHMMSS.pkl`
   - Trained Random Forest classifier

2. **Scaler**: `scaler_YYYYMMDD_HHMMSS.pkl`
   - StandardScaler for feature normalization

3. **Encoders**: `encoders_YYYYMMDD_HHMMSS.pkl`
   - Label encoders for categorical features

4. **Features**: `feature_names_YYYYMMDD_HHMMSS.json`
   - List of 33 feature names

**All 4 files must have matching timestamps!**

---

## 🔍 **Debug Mode**

### **Enable Verbose Logging:**

```bash
streamlit run app/streamlit_app.py --logger.level=debug
```

### **Check Streamlit Cache:**

```bash
# Clear cache if needed
streamlit cache clear
```

### **Check Python Path:**

```python
python -c "
import sys
from pathlib import Path
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('Current dir:', Path.cwd())
print('Parent dir:', Path.cwd().parent)
"
```

---

## 🎯 **Quick Test**

### **Minimal Test Script:**

```python
# test_app.py
from pathlib import Path
import sys

sys.path.append(str(Path.cwd() / 'src'))

try:
    from predict import HousePurchasePredictor
    print("✅ Imports work")
    
    models_dir = Path('models')
    model_files = list(models_dir.glob('best_model_*.pkl'))
    print(f"✅ Found {len(model_files)} model(s)")
    
    if model_files:
        latest = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"✅ Latest model: {latest.name}")
        
        parts = latest.stem.split('_')
        timestamp = '_'.join(parts[-2:])
        print(f"✅ Timestamp: {timestamp}")
        
        predictor = HousePurchasePredictor(
            model_path=str(latest),
            scaler_path=str(models_dir / f'scaler_{timestamp}.pkl'),
            encoders_path=str(models_dir / f'encoders_{timestamp}.pkl'),
            feature_names_path=str(models_dir / f'feature_names_{timestamp}.json')
        )
        print("✅ Model loaded successfully!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

Run: `python test_app.py`

---

## 💡 **Pro Tips**

1. **Always retrain after major changes**
   - Ensures all files have matching timestamps

2. **Keep one model version**
   - Delete old models to avoid confusion

3. **Use virtual environment**
   - Prevents dependency conflicts

4. **Check logs first**
   - Terminal shows detailed error messages

5. **Test with sample data**
   - Use provided defaults before custom inputs

---

## 🆘 **Still Having Issues?**

### **1. Clean Restart:**

```bash
# Stop the app (Ctrl+C)
# Clear cache
streamlit cache clear
# Restart
./run_app.sh
```

### **2. Fresh Model:**

```bash
# Delete all models
rm models/*.pkl models/*.json models/*.csv
# Retrain
python src/main_pipeline.py
# Try app again
./run_app.sh
```

### **3. Check Logs:**

Look for detailed error messages in terminal output.

### **4. Manual Test:**

```bash
# Test prediction module separately
cd src
python predict.py ../data/global_house_purchase_dataset.csv
```

---

## ✅ **Checklist Before Running**

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model trained (`python src/main_pipeline.py`)
- [ ] Model files exist in `models/` directory
- [ ] All 4 files have matching timestamps
- [ ] Port 8501 is available
- [ ] Current directory is project root

---

## 🎉 **Success Indicators**

When the app works correctly, you should see:

- ✅ "Model loaded successfully" (no errors)
- ✅ Sidebar shows model metrics
- ✅ Form inputs are all visible
- ✅ Predictions work instantly
- ✅ Gauge charts display properly
- ✅ Batch upload works

---

**The app is now working! Run `./run_app.sh` to start! 🚀**
