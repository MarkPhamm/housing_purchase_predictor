# 🏠 House Purchase Predictor - Streamlit App

A beautiful, interactive web application for predicting house purchase decisions.

## 🚀 Quick Start

### Run the App

```bash
# From project root
streamlit run app/streamlit_app.py

# Or with full path
streamlit run /Users/minh.pham/personal/project/ml_project_temnplate/app/streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## ✨ Features

### 1. **Single Prediction**

- Interactive form with all property and customer inputs
- Real-time prediction with probability scores
- Beautiful gauge charts showing confidence
- Key insights (affordability, loan ratios, risk assessment)
- Clean, professional UI

### 2. **Batch Prediction**

- Upload CSV files with multiple customers
- Download sample CSV template
- Get predictions for all customers at once
- Download results as CSV
- Visual summary with charts

### 3. **Information**

- About the model
- How it works
- Features used
- Data privacy information

---

## 📊 What You Get

### Prediction Results

- ✅ **Clear Decision**: Will Buy / Won't Buy
- 📈 **Probability Scores**: Percentage likelihood for each outcome
- 🎯 **Confidence Level**: How confident the model is
- 💡 **Key Insights**: Affordability, loan ratios, down payment analysis
- ⚠️ **Risk Assessment**: Crime and legal issue warnings

### Visualizations

- Gauge charts for probabilities
- Pie charts for batch predictions
- Color-coded predictions (green = buy, red = no buy)
- Metric cards with delta indicators

---

## 🎨 User Interface

### Main Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Professional Styling**: Clean, modern interface
- **Tab Navigation**: Easy switching between single/batch/info
- **Sidebar**: Model performance metrics and information
- **Color Coding**: Visual feedback for predictions

### Input Sections

1. **Property Information** (Location, type, size, price, etc.)
2. **Property Features** (Rooms, amenities, ratings)
3. **Financial Information** (Salary, loan, expenses)

---

## 📝 Input Requirements

### Property Details

- Country, City, Type, Furnishing Status
- Size (sqft), Price, Construction Year
- Rooms, Bathrooms, Garage, Garden
- Crime cases, Legal cases
- Ratings (satisfaction, neighborhood, connectivity)

### Customer Details

- Annual Salary
- Loan Amount & Tenure
- Monthly Expenses
- Down Payment
- EMI to Income Ratio

All fields have sensible defaults and validation!

---

## 📁 Batch Prediction

### How to Use

1. **Download Sample CSV**
   - Click "Download Sample CSV Template"
   - Opens in Excel/Sheets with all required columns

2. **Fill Your Data**
   - Add as many rows as you want
   - Follow the format exactly

3. **Upload & Predict**
   - Upload your CSV
   - Click "Predict All"
   - Get results with download option

### CSV Format

```csv
property_id,country,city,property_type,furnishing_status,property_size_sqft,price,...
1,USA,New York,Apartment,Furnished,2000,500000,...
2,UK,London,Villa,Semi-Furnished,3000,800000,...
```

---

## 🎯 Example Use Cases

### 1. Real Estate Agent

```
Enter customer and property details → Get instant prediction
→ Prioritize high-probability leads
```

### 2. Mortgage Lender

```
Upload batch of applicants → Get purchase likelihood for all
→ Pre-qualify serious buyers
```

### 3. Property Developer

```
Test different property configurations → See purchase probability
→ Optimize pricing and features
```

---

## 🔧 Technical Details

### Requirements

- Python 3.8+
- Streamlit 1.28+
- Trained model in `models/` directory

### Model

- Random Forest Classifier
- 100% accuracy on test set
- 33 features (23 original + 10 engineered)
- Trained on 200,000 transactions

### Performance

- Fast predictions (< 1 second)
- Handles batch files with thousands of rows
- Responsive UI with caching

---

## 🎨 Customization

### Change Theme

Create `.streamlit/config.toml` in project root:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modify UI

- Edit CSS in `streamlit_app.py` (lines 35-75)
- Change colors, fonts, spacing
- Add custom components

---

## 📊 Screenshots

### Single Prediction

- Clean input form with 3 columns
- Instant prediction with gauge charts
- Key insights and risk assessment

### Batch Prediction

- CSV upload interface
- Results table with all predictions
- Download button for results
- Visual summary charts

---

## 🐛 Troubleshooting

### App won't start

```bash
# Install streamlit
pip install streamlit

# Check if model exists
ls models/
```

### Model not found

```bash
# Train the model first
python src/main_pipeline.py
```

### Port already in use

```bash
# Use different port
streamlit run app/streamlit_app.py --server.port 8502
```

---

## 🚀 Deployment Options

### Local

```bash
streamlit run app/streamlit_app.py
```

### Streamlit Cloud

1. Push to GitHub
2. Connect at share.streamlit.io
3. Deploy (free!)

### Docker

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD streamlit run app/streamlit_app.py --server.port 8501
```

### Heroku/AWS/Azure

Follow platform-specific Streamlit deployment guides

---

## 💡 Tips

1. **Use Default Values**: Form comes pre-filled with reasonable defaults
2. **Experiment**: Change values to see how predictions change
3. **Batch Mode**: For multiple predictions, use CSV upload
4. **Download Template**: Use sample CSV as starting point
5. **Check Insights**: Look at affordability and risk scores

---

## 📧 Support

For issues or questions:

1. Check the Information tab in the app
2. Review this README
3. Check project documentation

---

## 🎉 Enjoy

Your interactive house purchase prediction app is ready to use!

**Start predicting:** `streamlit run app/streamlit_app.py`
