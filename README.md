# food-delivery-demand-analysis
# Food Delivery Demand Analysis

A Python-based data analysis project focused on understanding demand patterns, customer behavior, and operational trends in a food delivery system using synthetic order data.

## Overview
This project performs end-to-end exploratory data analysis (EDA) on food delivery orders, followed by basic predictive modeling and customer segmentation. The objective is to extract actionable insights related to order volume, peak demand periods, customer spending behavior, and delivery dynamics.

The analysis is structured to mirror a real-world analytics workflow, with an emphasis on clarity, interpretability, and practical insights rather than model complexity.

## Key Objectives
- Analyze temporal demand patterns across hours and weekdays
- Identify high-performing delivery areas and cuisines
- Segment customers based on ordering behavior
- Forecast short-term order demand using regression techniques
- Visualize relationships between operational and customer metrics

## Technical Implementation
- *Data Generation:* Synthetic dataset created using NumPy to simulate realistic food delivery orders
- *Data Processing & EDA:* Pandas used for data manipulation, aggregation, and statistical summaries
- *Visualization:* Matplotlib and Seaborn used for comparative and trend-based analysis
- *Customer Segmentation:* K-Means clustering applied to aggregated customer-level features
- *Demand Forecasting:* Linear Regression used to predict short-term order volume trends

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Analysis Highlights
- Identification of peak ordering hours and weekdays
- Comparison of order value distributions across different areas
- Cuisine popularity analysis
- Customer segmentation based on order frequency and average spend
- Seven-day demand forecast derived from historical order trends

## Design Decisions
- Synthetic data was used to maintain control over feature distributions and edge cases.
- Linear Regression was selected for forecasting to prioritize interpretability over model complexity.
- Customer segmentation focuses on order frequency and average spend, aligning with common business KPIs

