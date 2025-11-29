# CMSC_6950_Project: Climate Data Analysis Project (CMSC 6950)

## Data Source
Project - Climate Variability Analysis Using Daily Temperature Records 
Dataset: Daily Climate Observations, St. John’s West (Station 8403603), 2024.
Government of Canada  
https://climate.weather.gc.ca


## 1.0 Project Objective 
This project analyzes daily climate observations from the St. John's
West Weather Station (ID: 8403603) for the year 2024. The dataset is
obtained from the Government of Canada's Environment and Climate Change
(ECCC) Climate Data Portal.

The analysis explores:

-   Daily temperature patterns
-   Extreme temperature identification
-   Monthly summaries
-   Trend estimation with linear regression
-   Seasonal comparisons
-   Temperature heatmaps


## 2.0 Project Structure


CMSC_6950_Project/
|
|--src/
|   analysis.py           – main statistical functions used in the project  
│   data_processing.py    – script for cleaning and preparing the raw climate data  
│   compute.py            – simple console script that runs the core analysis  
│   figure_plot.py        – code for generating all plots and visualizations  
│
|-- data/
|    en_climate_daily_NL_8403603_2024_P1D.csv   – Raw downloaded weather dataset   
|    processed_en_climate_daily_NL_8403603_2024_P1D.csv   – cleaned dataset used for analysis  
|	 
|
|-- figures/
|     (all generated figures are saved here after running figure_plot.py)
|
|--tests/
|     test_analysis.py      – unit tests for analysis.py functions  
|
|--requirements.txt      – list of Python dependencies  
|--README.md             – project documentation 


## 3. Setup Instructions

# 3.1 Virtual Environment

WSL / Linux:

    python3 -m venv venv
    source venv/bin/activate

Windows PowerShell:

    python -m venv venv
    venv\Scripts\activate



# 3.2 Install required Python packages

pip install -r requirements.txt


## 4.0 Regenerating the Dataset

    python src/data_processing.py 

## 5.0 Generating Figures

Generate all:

    python src/figure_plot.py

Individual plots:

    python src/figure_plot.py --plot daily
    python src/figure_plot.py --plot extremes
    python src/figure_plot.py --plot monthly
    python src/figure_plot.py --plot trend
    python src/figure_plot.py --plot heatmap
    python src/figure_plot.py --plot season


## 6.0 Running Unit Tests

    pytest -v






