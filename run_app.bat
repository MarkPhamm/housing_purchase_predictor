@echo off
REM Launch script for House Purchase Predictor Streamlit App (Windows)

echo 🏠 Starting House Purchase Predictor App...
echo.
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

REM Check if streamlit is installed
where streamlit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Streamlit is not installed. Installing now...
    pip install streamlit
)

REM Check if model exists
if not exist "models\*.pkl" (
    echo ⚠️  No trained model found!
    echo Please run: python src/main_pipeline.py first
    exit /b 1
)

REM Run the app
streamlit run app/streamlit_app.py

