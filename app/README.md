# Mortality Risk Predictor - Interactive Dashboard

An interactive web application for exploring mortality risk predictions using the trained logistic regression model.

## Features

- **Temporal Controls**: Adjust season (month), day of week, and time of day for admission and discharge
- **Clinical Controls**: Select specialty, unit, diagnostic group, and ICD codes
- **Stay Duration**: Adjust hospital and unit length of stay with sliders
- **Real-time Predictions**: See mortality risk probability updated instantly as you adjust parameters

## Running the App

### Option 1: Using the run script
```bash
./run_app.sh
```

### Option 2: Direct Streamlit command
```bash
streamlit run app/app.py
```

### Option 3: From project root
```bash
cd /path/to/mortality-modeling
source env/bin/activate  # if using virtual environment
streamlit run app/app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Requirements

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Adjust the sliders and dropdowns to set patient parameters
2. The mortality risk probability updates automatically
3. View the risk level (High/Low) and model prediction
4. Expand "Feature Summary" to see all selected parameters
