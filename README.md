# Forecasting US Employment Growth with an AR(3) Model

This project applies an Autoregressive model of order three (AR(3)) to US employment data to analyze year-on-year employment growth and produce 24-month-ahead forecasts.
The project was developed for the Macroeconometrics course at the University of Palermo.

## Objectives

- Analyze the dynamics of US employment growth.
- Estimate an AR(3) model using Ordinary Least Squares (OLS).
- Express the model in companion matrix form.
- Assess stationarity through the characteristic roots.
- Compute recursive multi-step forecasts.
- Construct forecast intervals using the Root Mean Squared Forecast Error (RMSFE).

## Data

The analysis uses monthly US employment data from January 1974 to December 2025.
The year-on-year employment growth rate is calculated as:

$$
Growth_t = \left(\frac{Employment_t}{Employment_{t-12}} - 1\right) \times 100
$$

The first 12 observations are lost when constructing the growth rate.

## Methodology

The estimated model is:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \phi_3 y_{t-3} + u_t
$$

The model is estimated using matrix-based OLS:

$$
\hat{\beta} = (X'X)^{-1}X'y
$$

The AR(3) is then written in companion form to obtain recursive forecasts.

## Results

The estimated AR(3) indicates substantial persistence in US employment growth.
The model is used to generate forecasts for the 24 months following the end of the estimation sample, together with 68%, 90% and 95% forecast intervals.
The full estimation results, mathematical derivations and discussion are available in the accompanying report.

## Project Structure

```text
Employment_Growth_AR_Model/
├── main.py
├── PAYEMS.csv
├── employment_growth_plot.png
├── forecast.png
├── report/
│   └── report.pdf
├── .gitignore
└── README.md
