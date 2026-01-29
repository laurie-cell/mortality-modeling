#!/bin/bash

# Activate virtual environment if it exists
if [ -d "env" ]; then
    source env/bin/activate
fi

# Run the Streamlit app
streamlit run app/app.py
