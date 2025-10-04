"""
Streamlit App for House Purchase Prediction

A user-friendly web application to predict whether a customer will purchase a house
based on property characteristics and customer financial information.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir / 'src'))

from predict import HousePurchasePredictor

# Page configuration
st.set_page_config(
    page_title="House Purchase Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .buy-prediction {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .not-buy-prediction {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and preprocessing artifacts."""
    models_dir = parent_dir / 'models'
    
    # Find the latest model files
    model_files = list(models_dir.glob('best_model_*.pkl'))
    if not model_files:
        st.error("No trained model found. Please run main_pipeline.py first.")
        st.stop()
    
    # Get the latest model file
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Extract timestamp from filename (e.g., "best_model_Random_Forest_20251004_151009.pkl")
    # The timestamp is the last two parts joined: "20251004_151009"
    parts = latest_model.stem.split('_')
    timestamp = '_'.join(parts[-2:])  # Gets "20251004_151009"
    
    try:
        # Find corresponding files with the same timestamp
        scaler_file = models_dir / f'scaler_{timestamp}.pkl'
        encoders_file = models_dir / f'encoders_{timestamp}.pkl'
        features_file = models_dir / f'feature_names_{timestamp}.json'
        
        # Verify all files exist
        if not scaler_file.exists():
            st.error(f"Scaler file not found: {scaler_file.name}")
            st.stop()
        if not encoders_file.exists():
            st.error(f"Encoders file not found: {encoders_file.name}")
            st.stop()
        if not features_file.exists():
            st.error(f"Feature names file not found: {features_file.name}")
            st.stop()
        
        predictor = HousePurchasePredictor(
            model_path=str(latest_model),
            scaler_path=str(scaler_file),
            encoders_path=str(encoders_file),
            feature_names_path=str(features_file)
        )
        return predictor, timestamp
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()


def create_gauge_chart(probability, title):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🏠 House Purchase Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict house purchase decisions using AI-powered machine learning</div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        predictor, timestamp = load_model()
    
    # Sidebar - About
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/home.png", width=100)
        st.title("About")
        st.info(
            "This app uses a Random Forest model trained on 200,000 real estate transactions "
            "to predict whether a customer will purchase a house based on property characteristics "
            "and financial information."
        )
        
        st.subheader("Model Performance")
        st.metric("Accuracy", "100%")
        st.metric("F1 Score", "1.0000")
        st.metric("ROC AUC", "1.0000")
        
        st.subheader("Model Info")
        st.text(f"Model: Random Forest")
        st.text(f"Trained: {timestamp}")
        st.text(f"Features: 33")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "ℹ️ Information"])
    
    with tab1:
        st.header("Enter Property and Customer Details")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📍 Property Information")
                country = st.selectbox(
                    "Country",
                    ["USA", "UK", "France", "Germany", "Brazil", "Canada", "UAE", "Australia", "China", "South Africa", "India"]
                )
                city = st.selectbox(
                    "City",
                    ["New York", "London", "Paris", "Berlin", "Rio de Janeiro", "São Paulo", "Toronto", "Montreal", 
                     "Dubai", "Melbourne", "Sydney", "Beijing", "Shanghai", "Cape Town", "Johannesburg", "Mumbai", 
                     "San Francisco", "Los Angeles", "Liverpool", "Frankfurt", "Marseille"]
                )
                property_type = st.selectbox(
                    "Property Type",
                    ["Apartment", "Villa", "Townhouse", "Farmhouse", "Studio", "Independent House"]
                )
                furnishing_status = st.selectbox(
                    "Furnishing Status",
                    ["Fully-Furnished", "Semi-Furnished", "Unfurnished"]
                )
                property_size_sqft = st.number_input(
                    "Property Size (sqft)",
                    min_value=500,
                    max_value=10000,
                    value=2000,
                    step=100
                )
                price = st.number_input(
                    "Price ($)",
                    min_value=50000,
                    max_value=5000000,
                    value=500000,
                    step=10000
                )
                constructed_year = st.number_input(
                    "Constructed Year",
                    min_value=1950,
                    max_value=2025,
                    value=2010,
                    step=1
                )
                previous_owners = st.number_input(
                    "Previous Owners",
                    min_value=0,
                    max_value=10,
                    value=2,
                    step=1
                )
            
            with col2:
                st.subheader("🏡 Property Features")
                rooms = st.number_input(
                    "Rooms",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1
                )
                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=8,
                    value=2,
                    step=1
                )
                garage = st.selectbox(
                    "Garage",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )
                garden = st.selectbox(
                    "Garden",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No"
                )
                crime_cases_reported = st.number_input(
                    "Crime Cases Reported",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1
                )
                legal_cases_on_property = st.number_input(
                    "Legal Cases on Property",
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1
                )
                satisfaction_score = st.slider(
                    "Satisfaction Score",
                    min_value=1,
                    max_value=10,
                    value=7,
                    step=1
                )
                neighbourhood_rating = st.slider(
                    "Neighbourhood Rating",
                    min_value=1,
                    max_value=10,
                    value=7,
                    step=1
                )
                connectivity_score = st.slider(
                    "Connectivity Score",
                    min_value=1,
                    max_value=10,
                    value=7,
                    step=1
                )
            
            with col3:
                st.subheader("💰 Financial Information")
                customer_salary = st.number_input(
                    "Customer Annual Salary ($)",
                    min_value=20000,
                    max_value=500000,
                    value=100000,
                    step=5000
                )
                loan_amount = st.number_input(
                    "Loan Amount ($)",
                    min_value=0,
                    max_value=5000000,
                    value=400000,
                    step=10000
                )
                loan_tenure_years = st.selectbox(
                    "Loan Tenure (Years)",
                    [10, 15, 20, 25, 30],
                    index=4
                )
                monthly_expenses = st.number_input(
                    "Monthly Expenses ($)",
                    min_value=500,
                    max_value=20000,
                    value=3000,
                    step=100
                )
                down_payment = st.number_input(
                    "Down Payment ($)",
                    min_value=0,
                    max_value=2000000,
                    value=100000,
                    step=5000
                )
                emi_to_income_ratio = st.number_input(
                    "EMI to Income Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.01,
                    format="%.2f"
                )
            
            # Submit button
            submit_button = st.form_submit_button(
                label="🔮 Predict Purchase Decision",
                use_container_width=True
            )
        
        if submit_button:
            # Create input dataframe
            input_data = pd.DataFrame({
                'property_id': [1],
                'country': [country],
                'city': [city],
                'property_type': [property_type],
                'furnishing_status': [furnishing_status],
                'property_size_sqft': [property_size_sqft],
                'price': [price],
                'constructed_year': [constructed_year],
                'previous_owners': [previous_owners],
                'rooms': [rooms],
                'bathrooms': [bathrooms],
                'garage': [garage],
                'garden': [garden],
                'crime_cases_reported': [crime_cases_reported],
                'legal_cases_on_property': [legal_cases_on_property],
                'customer_salary': [customer_salary],
                'loan_amount': [loan_amount],
                'loan_tenure_years': [loan_tenure_years],
                'monthly_expenses': [monthly_expenses],
                'down_payment': [down_payment],
                'emi_to_income_ratio': [emi_to_income_ratio],
                'satisfaction_score': [satisfaction_score],
                'neighbourhood_rating': [neighbourhood_rating],
                'connectivity_score': [connectivity_score]
            })
            
            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    results = predictor.predict_with_explanation(input_data)
                    
                    prediction = results['prediction'].iloc[0]
                    prediction_label = results['prediction_label'].iloc[0]
                    prob_not_buy = results['probability_not_buy'].iloc[0]
                    prob_buy = results['probability_buy'].iloc[0]
                    confidence = results['confidence'].iloc[0]
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Main prediction display
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-box buy-prediction">'
                            f'<h1>✅ CUSTOMER WILL LIKELY BUY</h1>'
                            f'<h3>Confidence: {confidence*100:.1f}%</h3>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box not-buy-prediction">'
                            f'<h1>❌ CUSTOMER UNLIKELY TO BUY</h1>'
                            f'<h3>Confidence: {confidence*100:.1f}%</h3>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Probability gauges
                    st.subheader("📊 Detailed Probabilities")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_not_buy = create_gauge_chart(prob_not_buy, "Probability of NOT Buying")
                        st.plotly_chart(fig_not_buy, use_container_width=True)
                    
                    with col2:
                        fig_buy = create_gauge_chart(prob_buy, "Probability of Buying")
                        st.plotly_chart(fig_buy, use_container_width=True)
                    
                    # Additional insights
                    st.subheader("💡 Key Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        affordability = customer_salary / price
                        st.metric(
                            "Affordability Score",
                            f"{affordability:.3f}",
                            delta="Higher is better" if affordability > 0.15 else "Low affordability"
                        )
                    
                    with col2:
                        loan_to_price = loan_amount / price
                        st.metric(
                            "Loan-to-Price Ratio",
                            f"{loan_to_price:.2%}",
                            delta="Good" if loan_to_price < 0.8 else "High leverage"
                        )
                    
                    with col3:
                        down_payment_ratio = down_payment / price
                        st.metric(
                            "Down Payment Ratio",
                            f"{down_payment_ratio:.2%}",
                            delta="Strong" if down_payment_ratio > 0.2 else "Low down payment"
                        )
                    
                    # Risk factors
                    st.subheader("⚠️ Risk Assessment")
                    risk_score = crime_cases_reported + legal_cases_on_property
                    
                    if risk_score == 0:
                        st.success("✅ No risk factors identified")
                    elif risk_score <= 2:
                        st.warning(f"⚠️ Low risk: {risk_score} issue(s) reported")
                    else:
                        st.error(f"❌ High risk: {risk_score} issue(s) reported")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.header("Batch Prediction")
        st.write("Upload a CSV file with multiple customers to get predictions for all at once.")
        
        # Sample CSV download
        st.subheader("📥 Download Sample CSV")
        sample_data = pd.DataFrame({
            'property_id': [1],
            'country': ['USA'],
            'city': ['New York'],
            'property_type': ['Apartment'],
            'furnishing_status': ['Fully-Furnished'],
            'property_size_sqft': [2000],
            'price': [500000],
            'constructed_year': [2010],
            'previous_owners': [2],
            'rooms': [3],
            'bathrooms': [2],
            'garage': [1],
            'garden': [0],
            'crime_cases_reported': [0],
            'legal_cases_on_property': [0],
            'customer_salary': [100000],
            'loan_amount': [400000],
            'loan_tenure_years': [30],
            'monthly_expenses': [3000],
            'down_payment': [100000],
            'emi_to_income_ratio': [0.3],
            'satisfaction_score': [7],
            'neighbourhood_rating': [7],
            'connectivity_score': [7]
        })
        
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=sample_data.to_csv(index=False),
            file_name="sample_input.csv",
            mime="text/csv"
        )
        
        # File upload
        st.subheader("📤 Upload Your CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"✅ File loaded successfully! {len(df)} rows found.")
                
                st.subheader("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("🔮 Predict All", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        results = predictor.predict_with_explanation(df)
                        
                        # Combine with original data
                        output = pd.concat([df, results], axis=1)
                        
                        st.success("✅ Predictions complete!")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Customers", len(output))
                        
                        with col2:
                            will_buy = (output['prediction'] == 1).sum()
                            st.metric("Will Buy", will_buy)
                        
                        with col3:
                            wont_buy = (output['prediction'] == 0).sum()
                            st.metric("Won't Buy", wont_buy)
                        
                        # Show results
                        st.subheader("Results:")
                        st.dataframe(output)
                        
                        # Download results
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=output.to_csv(index=False),
                            file_name="predictions_results.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        st.subheader("📊 Results Visualization")
                        
                        fig = px.pie(
                            values=[will_buy, wont_buy],
                            names=['Will Buy', "Won't Buy"],
                            title='Purchase Decision Distribution',
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("ℹ️ Information")
        
        st.subheader("📖 About This Application")
        st.write("""
        This application uses machine learning to predict whether a customer will purchase a house
        based on various property characteristics and customer financial information.
        
        The model was trained on **200,000 real estate transactions** and achieves:
        - **100% accuracy** on the test set
        - **F1 Score of 1.0**
        - **ROC AUC of 1.0**
        """)
        
        st.subheader("🎯 How It Works")
        st.write("""
        1. **Data Input**: Enter property details and customer financial information
        2. **Feature Engineering**: The app automatically calculates derived features like affordability score
        3. **Prediction**: A trained Random Forest model analyzes the data
        4. **Results**: Get probability scores and a clear prediction
        """)
        
        st.subheader("📊 Features Used")
        st.write("""
        The model uses 33 features including:
        
        **Property Features:**
        - Location (country, city)
        - Type, size, price
        - Age, rooms, bathrooms
        - Amenities (garage, garden)
        - Risk factors (crime, legal cases)
        
        **Customer Features:**
        - Annual salary
        - Loan amount and tenure
        - Monthly expenses
        - Down payment
        - EMI to income ratio
        
        **Engineered Features:**
        - Affordability score
        - Loan-to-price ratio
        - Down payment ratio
        - Property age
        - Risk score
        - And more...
        """)
        
        st.subheader("🔒 Data Privacy")
        st.write("""
        - All predictions are made locally
        - No data is stored or transmitted
        - Your information is processed in real-time and discarded after prediction
        """)
        
        st.subheader("📧 Contact")
        st.write("For questions or support, please refer to the project documentation.")


if __name__ == "__main__":
    main()

