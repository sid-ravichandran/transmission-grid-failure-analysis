import streamlit as st
import pandas as pd
import numpy as np
from app_functions import (load_data, process_data, eda_faults_counts_barplot, eda_faults_duration_barplot,
                           overall_failure_rates, overall_failure_rate_modelling, asset_failure_rates, asset_failures_recent_trend, gamma_fit,
                           repair_time_histograms, model_distribution_fits)

# Page config
st.set_page_config(
    page_title="Transmission Asset Failure Analysis",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ Transmission Asset Failure Analysis")
st.markdown("### Generation of Monte Carlo Input Parameters")
st.markdown("*Brief Analysis Report of historic failure event data of Feeder Assets for probabilistic risk forecasting using National Grid's datasets*")
st.write("**Author:** Siddharth Ravichandran, Feb 2026")
st.write("www.linkedin.com/in/sid-ravichandran")
st.write('------------------------------------------------------------------------')

# Sidebar
st.sidebar.header("Report Navigation")
section = st.sidebar.radio("Jump to section:", [
    "Executive Summary",
    "Approach Followed and Limitations",
    "Exploratory Data Analysis",
    "Overall Failure Rate", 
    "Failure Rate due to Asset Deterioration",
    "Repair Time Modelling",
    "Scaling to Real Operational Datasets"
])


# =====================================
# EXECUTIVE SUMMARY
# =====================================
if section == "Executive Summary":
    st.header("Executive Summary")
    
    st.markdown("""
    ### Key Findings
    
    1. **Overall failure rate per asset-km** shows linear increasing trend (+0.0027/km per year) with value of 0.154 in 2024, and a Uniformly distributed uncertainty with bounds ()
    2. **Deterioration related failure rate relatively stable over past 10 years** with mean value at 0.059/km and can be modelled using a Gamma distribution (shape=45.520, loc=0.000, scale=0.001)
    3. **Repair times** follow uniform distribution (60-840 mins) with discrete values rounded to nearest hour
    
    ### Monte Carlo Input Summary
    
    | Parameter | Distribution | Notes |
    |-----------|-------------|-------|
    | Overall Failure Rate | Linearly increasing rate with Uniform (-0.0010, 0.0010) uncertainty (/km) | Linear trend projected |
    | Deterioration Failure Rate | Gamma(shape=45.520, loc=0.000, scale=0.001) | Recent 10-year stable average |
    | Repair Time | Uniform(60, 840) mins | Discrete, rounded to nearest hour |
    """)

    st.markdown("### Likelihood of failure per year of power cuts")
    st.write("To calculate the likelihood of failure per year of power cuts, we can use the failure rate per km per year as per the above, and the total length of the feeder assets. The formulae are as follows:")
    st.write("**Step 1:** Calculate Expected Number of Failures (λ)")
    st.latex(r"\text{λ} = \text{Overall Failure Rate (per km/year)} \times \text{Total Asset Length (km)}")

    st.write("**Step 2:** Convert to Probability Using Poisson Distribution")
    st.write("The number of failures can be assumed to follow a Poisson distribution with parameter λ")
    st.latex(r"P(X = k) = \frac{e^{-\lambda} . \lambda^k}{k!}")
    st.latex(r"P(X \geq 1) = 1 - P(X = 0) = 1 - e^{-\lambda}")


# =====================================
# Approach Followed
# =====================================
if section == "Approach Followed and Limitations":
    st.header("Approach Followed and Limitations")

    st.markdown("""
    The approach followed in this analysis can be summarised as follows:
    1. **Data Loading and Preprocessing**: The raw data was loaded and processed to create a clean dataset for analysis. This involved handling missing values and timestamps, categorising fault descriptions based on EDA, and calculating relevant metrics such as fault duration per km.
    2. **Exploratory Data Analysis (EDA)**: I performed EDA to understand the distribution of fault types, their durations/km, and how they have evolved over time. This helped me identify trends and patterns in the data.
    3. **Failure Rate Analysis**: I calculated the overall failure rate and the failure rate due to asset deterioration. I also fitted probability distributions to these rates to model their variability and uncertainty.
    4. **Repair Time Modelling**: I analysed the distribution of repair times for different fault types and fitted appropriate probability distributions to model the repair times for use in the Monte Carlo simulation.
    5. **Monte Carlo Input Generation**: Based on the analyses, I generated input parameters for the Monte Carlo simulation, including the overall failure rate, deterioration failure rate, and repair time distributions.
    """)

    st.markdown("""
    **Implementation Details:** The analysis was implemented in Python using the following libraries:
    - Pandas and Numpy for data manipulation
    - Plotly and Seaborn for visualization
    - SciPy for statistical analysis
    - Streamlit for creating the interactive report
    """)

    st.markdown("""
    **Limitations and Assumptions:**
    - The analysis is based on a small sample dataset, hence the processing was done using local processing on Pandas and not distributed computing frameworks like PySpark or Dask.
    - In a production setup, more details would be required for digging into the unknown causes and more granular fault categorisation will be helpful for accuracy
    - When categorising faults as Asset Deterioration related vs Third Party, it has been assumed that 20% of the unknown causes are due to asset deterioration, and the rest are due to third party / external causes. This is a simplifying assumption and may not be accurate, but it allows us to proceed with the analysis given the limitations of the data.
    """)


# =====================================
# EDA
# =====================================
df_raw, total_asset_length = load_data()
df = process_data(df_raw)

if section == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    st.markdown("#### Distribution of Fault Descriptions")
    st.write("The bar plot below shows the distribution of different fault descriptions in the dataset. This helps us understand the most common causes of feeder asset failures.")
    st.markdown("- '**Duplicate trip record from control system**' will be excluded as it is a duplicate and these records do not add any new information")

    st.write("The fault descriptions can be categorised as follows:")
    st.markdown("- **Unknown Causes:**: 'HV trip - limited historical detail', 'HV trip - cause unclear, no defects found', 'Unknown cause - reset successful'")
    st.markdown("- **Asset Deterioration Related:** 'Conductor damage mid span', 'Insulator failure reported on cross-arm', 'UG cable insulation breakdown', 'Cable joint failure suspected'")
    st.markdown("- **External / third part related:**:  All others, for example 'Tree contact', 'Lightning strike', 'Animal contact', 'Vehicle collision', etc.")

    fig = eda_faults_counts_barplot(df_raw)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Fault Duration per km by Fault Description")
    st.write("Since the length of the feeder assets can vary significantly, we normalise the average fault duration per km to get a more accurate picture of the impact of different fault types.")
    st.write("The below charts show that while the **TOTAL** fault duration per km shows significant variation (2000-6000 min/km), the **AVERAGE** fault duration per km is more consistent across different fault descriptions (35-40 min/km).")

    fig1, fig2 = eda_faults_duration_barplot(df)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)


# =====================================
# Overall Failure Rate
# =====================================
if section == "Overall Failure Rate":
    st.header("Overall Failure Rate")

    st.markdown("#### Overall Failure Rate Calculation")
    st.write("The overall failure rate is calculated as the total number of faults per km of feeder asset per year. This metric helps us understand the frequency of failures relative to the length of the assets and time, which is crucial for risk forecasting and maintenance planning.")
    st.write("The plot below shows the **increasing trend** of the overall failure rate over time. An increasing trend indicates deteriorating asset conditions or emerging risks that need to be addressed.")
    
    df_failure_rates, fig = overall_failure_rates(df, total_asset_length)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Modelling failure rate per km for a Monte Carlo simulation")
    st.write("To model the failure rate per km for a Monte Carlo simulation, we will need to fit a probability distribution to the historical failure rate data. This allows us to capture the variability and uncertainty in the failure rates, which is essential for generating realistic scenarios in the simulation.")
    st.write("Since this looks like a linearly increasing trend, the approach to model this will be - **Linear Projection for Mean with a Distribution fitted to the Residual Uncertainty**")

    st.write("The plot below shows the fitted overall failure rate model, which can be used as an input for the Monte Carlo simulation to generate future failure rate scenarios based on historical trends and variability.")
    fig1, fig2, residuals = overall_failure_rate_modelling(df_failure_rates)
    st.pyplot(fig1, width='content')

    st.write("**An observation**: There is a regular pattern in the residuals w.r.t. mean every 5 years, which could be due to a 5-yearly maintenance cycle that causes a temporary improvement in failure rates after maintenance activities are performed. This pattern should be taken into account when interpreting the results and making projections based on the model.")
    st.write(f"Analysing the goodness of fit of the Residuals against the different distributions considered, it seems that a **Uniform Distribution** with bounds of {min(residuals):.4f} and {max(residuals):.4f} provides the best fit to the residuals, which represent the uncertainty around the linear trend. This distribution can be used in the Monte Carlo simulation to model the variability in failure rates around the projected linear trend.")
    st.pyplot(fig2, width='content')


# =====================================
# Failure Rate due to Asset Deterioration
# =====================================
if section == "Failure Rate due to Asset Deterioration":
    st.header("Failure Rate due to Asset Deterioration")

    st.markdown("#### Failure Rate Calculation due to Asset Deterioration")
    st.write("The failure rate due to asset deterioration is calculated as the number of faults attributed to asset deterioration per km of feeder asset per year. This metric helps us understand the contribution of asset aging and wear to overall system failures.")
    st.write("Note that a contribution factor of 0.2 was applied to the faults attributed to unknown causes, since some of these may have been due to asset deterioration but lacked sufficient information for accurate classification.")
    st.write("The plot below shows a **initially increasing trend** of the failure rate over time but **relatively stable over the last 10 years**.")

    df_asset_failure_rates, fig = asset_failure_rates(df, total_asset_length)
    st.plotly_chart(fig, use_container_width=True)

    st.write("The recent historical data of the past 10 years is more relevant than the older data and seems like a good dataset (albeit sparse) for fitting a probability distribution to model the failure rate due to asset deterioration for the Monte Carlo simulation.")

    fig1, fig2 = asset_failures_recent_trend(df_asset_failure_rates)
    st.pyplot(fig1, width='content')
    st.pyplot(fig2, width='content')

    st.write("The data can be seen to be slightly skewed to the right, which makes the **Gamma distribution** a suitable candidate. This is also borne out by the goodness of fit tests against the different distributions considered")

    fig = gamma_fit(df_asset_failure_rates)
    st.pyplot(fig, width='content')


# =====================================
# Repair Time Modelling
# =====================================
if section == "Repair Time Modelling":
    st.header("Repair Time Modelling")

    st.markdown("### Distributions of Time to Repair for All Fault Types and Asset Deterioration Related Faults")
    st.write("The time to repair for all fault types and asset deterioration related faults can be modelled using probability distributions. This allows us to capture the variability in repair times, which is crucial for accurate risk forecasting and resource planning.")
    st.write("The following observations can be made from the distribution of time to repair:")
    st.markdown("- Both distributions are quite 'flat' over a wide range of repair times (60 - 840 minutes), which initially suggests that a uniform distribution may be a good fit for modelling repair times.")
    st.markdown("- The discrete bars suggest that the repair times are rounded to the nearest hour, which is common in operational data and should be taken into account when fitting distributions.")

    fig1, fig2 = repair_time_histograms(df)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Choosing a Probability Distribution for Modelling Repair Times")
    st.write("To choose a probability distribution for modelling repair times, we can perform goodness-of-fit tests and visual assessments such as Q-Q plots. The goal is to identify a distribution that closely matches the empirical data, allowing us to generate realistic repair time scenarios in the Monte Carlo simulation.")
    st.write("Consider the overall fault type data. 4 distributions were looked at for evaluating fits - Normal, Uniform, Gamma, and Lognormal. The Q-Q plots show that the **uniform distribution provides the best fit to the repair time data**, as the points in the Q-Q plot for the uniform distribution are closest to the reference line compared to the other distributions.")
    fig = model_distribution_fits(df['fault_duration_minutes'])
    st.pyplot(fig, width='content')

    st.write("Based on the analysis, we can conclude that the uniform distribution is the most appropriate choice for modelling repair times in this context.")
    st.write("**Properties:** Min (a) = 60 minutes, Max (b) = 840 minutes, Results to be rounded to nearest hour in Monte Carlo simulation")


# =====================================
# Scaling to Real Operational Datasets
# =====================================
if section == "Scaling to Real Operational Datasets":
    st.header("Scaling to Real Operational Datasets")

    st.markdown("### Considerations for Scaling to Real Operational Datasets")
    st.write("When scaling the analysis to real operational datasets, several considerations need to be taken into account to ensure the accuracy and relevance of the results:")
    st.markdown("**Data Quality and Completeness**: Real operational datasets may have issues with data quality, such as missing values, inconsistencies, or inaccuracies. It is crucial to perform thorough data cleaning and preprocessing to address these issues before analysis.")
    st.markdown("**Computational Efficiency**: Real operational datasets can be significantly larger than the sample dataset used in this analysis. It is important to optimize the code and use efficient algorithms to handle larger datasets without compromising performance. Examples of tools that could be used include:")
    st.markdown("- Distributed computing frameworks like **PySpark** or **Dask**")
    st.markdown("- Cloud-based platforms like **Databricks** or **Snowflake** that can provide scalable resources for data processing and analysis.")
    st.markdown("- Data pipelines to automate the **ETL** (Extract, Transform, Load) processing and analysis workflow, ensuring that the analysis can run on a regular schedule and can be easily updated as new data becomes available.")
    st.markdown("**Model Validation**: The models and distributions fitted to the sample dataset should be validated against the real operational dataset to ensure that they still provide a good fit and accurately capture the underlying patterns and trends.")
    st.markdown("- Updates to parameters like the Monte Carlo distribution parameters will need to be regularly updated based on new dats, with more weightage given to more recent data to ensure relevance.")
    st.markdown("- Tools such as **ML flow** could be used to deploy models and track performance and drift in real time")
    st.markdown("**Domain Expertise**: Collaborating with domain experts is essential to interpret the results correctly and to ensure that the assumptions made in the analysis are valid in the context of real operational data.")
    st.markdown("**Continuous Monitoring and Updating**: As new data becomes available, it is important to continuously monitor the performance of the models and update them as necessary to maintain their accuracy and relevance over time.")
