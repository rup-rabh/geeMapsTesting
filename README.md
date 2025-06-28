# Satellite Image Analysis App

A simple Streamlit application for analyzing satellite imagery using different spectral indices (NDVI, NDSII, NDWI, MNDWI) for airports worldwide.

## Features

- Airport selection from a comprehensive global database
- Multiple spectral indices analysis options
- Adjustable threshold values
- Side-by-side image comparison

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Select an airport from the dropdown menu in the sidebar
2. Choose a spectral index (NDVI, NDSII, NDWI, or MNDWI)
3. Adjust the threshold value using the slider
4. Click "Submit" to analyze the images

Note: Currently using demo images. Integration with actual satellite imagery backend is pending. 