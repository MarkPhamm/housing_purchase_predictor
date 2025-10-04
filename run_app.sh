#!/bin/bash

# Launch script for House Purchase Predictor Streamlit App

echo "🏠 Starting House Purchase Predictor App..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed. Installing now..."
    pip install streamlit
fi

# Check if model exists
if [ ! -d "models" ] || [ -z "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    echo "⚠️  No trained model found!"
    echo "Please run: python src/main_pipeline.py first"
    exit 1
fi

# Run the app
streamlit run app/streamlit_app.py

