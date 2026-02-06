import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, gamma, probplot, norm

######################### Load and Process Data #########################
@st.cache_data
def load_data():
    """ Load Input CSV's and merge """

    df1 = pd.read_csv('feeder_attributes_full.csv')
    df2 = pd.read_csv('feeder_trip_events_full.csv')
    df = df2.merge(df1, on='feeder_id', how='left')
    df['asset_length_km'] = df['overhead_length_km'] + df['cable_length_km']
    total_asset_length = df1['overhead_length_km'].sum() + df1['cable_length_km'].sum()

    return df, total_asset_length


@st.cache_data
def process_data(df_raw):
    """ Cleaning and Feature Engineering of loaded data """

    df = df_raw[df_raw['fault_description'] != 'Duplicate trip record from control system'].copy()
    df['trip_start_time'] = pd.to_datetime(df['trip_start_time'])
    df['trip_end_time'] = pd.to_datetime(df['trip_end_time'])
    df['year'] = df['trip_start_time'].dt.year
    df['month'] = df['trip_start_time'].dt.month
    df['fault_duration_minutes'] = (df['trip_end_time'] - df['trip_start_time']).dt.total_seconds() / 60
    df['fault_duration_minutes_per_km'] = df['fault_duration_minutes'] / df['asset_length_km']
    return df


######################### EDA Plots #########################
def categorize_cause(fault_desc):
    asset_failure_causes = ['Conductor damage mid span', 'Insulator failure reported on cross-arm', 'UG cable insulation breakdown', 'Cable joint failure suspected']
    unknown_causes = ['HV trip – limited historical detail', 'HV trip – cause unclear, no defects found', 'Unknown cause – reset successful']

    if fault_desc in asset_failure_causes:
        return 'Asset Deterioration Related'
    elif fault_desc in unknown_causes:
        return 'Unknown Causes'
    else:
        return 'Third Party Related'


@st.cache_data        
def eda_faults_counts_barplot(df):
    """ Barplot of fault descriptions """

    cause_counts = Counter(df['fault_description'])
    cause_df = pd.DataFrame(cause_counts.items(), columns=['Fault Description', 'Count'])

    # create a column of cause categories
    cause_df['Cause Category'] = cause_df['Fault Description'].apply(categorize_cause)

    fig = px.bar(cause_df, x='Fault Description', y='Count', color='Cause Category', title='Distribution of Fault Descriptions')
    return fig


@st.cache_data
def eda_faults_duration_barplot(df):
    """ Barplot of average fault duration per km by fault description """

    df_faults_durations_sum = df[['fault_duration_minutes_per_km', 'fault_description']].groupby('fault_description').sum().reset_index()
    df_faults_durations_sum = df_faults_durations_sum.sort_values(by='fault_duration_minutes_per_km', ascending=False)

    # create a column of cause categories
    df_faults_durations_sum['Cause Category'] = df_faults_durations_sum['fault_description'].apply(categorize_cause)
    df_faults_durations_sum = df_faults_durations_sum.rename(columns={'fault_duration_minutes_per_km': 'Total Fault Duration per km', 'fault_description': 'Fault Description'})

    df_faults_durations_mean = df[['fault_duration_minutes_per_km', 'fault_description']].groupby('fault_description').mean().reset_index()
    df_faults_durations_mean = df_faults_durations_mean.sort_values(by='fault_duration_minutes_per_km', ascending=False)
    df_faults_durations_mean['Cause Category'] = df_faults_durations_mean['fault_description'].apply(categorize_cause)
    df_faults_durations_mean = df_faults_durations_mean.rename(columns={'fault_duration_minutes_per_km': 'Average Fault Duration per km', 'fault_description': 'Fault Description'})

    fig1 = px.bar(df_faults_durations_sum, x='Fault Description', y='Total Fault Duration per km', color='Cause Category', title='Total Fault Duration per km by Fault Description')
    fig2 = px.bar(df_faults_durations_mean, x='Fault Description', y='Average Fault Duration per km', color='Cause Category', title='Average Fault Duration per km by Fault Description')
    return fig1, fig2


@st.cache_data
def model_distribution_fits(input_data):
    """ Fit probability distributions to repair times for all fault types and asset deterioration related faults for use in Monte Carlo simulation """

    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    probplot(input_data, sparams=(1,), dist="lognorm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q Plot vs Lognormal')

    probplot(input_data, sparams=(1,), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot vs Normal')

    probplot(input_data, sparams=(1,), dist="uniform", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot vs Uniform')

    probplot(input_data, sparams=(1,), dist="gamma", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot vs Gamma')

    plt.tight_layout()
    # scale down font sizes in plot and dot size in Q-Q plot
    for ax in axes.flat:
        for line in ax.get_lines():
            line.set_markersize(3)
        ax.title.set_fontsize(6)
        ax.xaxis.label.set_fontsize(6)
        ax.yaxis.label.set_fontsize(6)
        ax.legend(fontsize=5)
        ax.tick_params(axis='both', which='major', labelsize=5)

    return fig

############################ Failure Rate Analysis #########################
@st.cache_data
def overall_failure_rates(df, total_asset_length):
    """ Calculate overall failure rates and create barplot of trend of failure rates over time """

    failure_rates = {
        'year': [],
        'Failure Rate per km': []
    }

    for y in sorted(list(df['year'].unique())):
        df_year = df[df['year'] == y]
        num_failures = len(df_year)
        failure_rate = num_failures / total_asset_length
        failure_rates['year'].append(y)
        failure_rates['Failure Rate per km'].append(failure_rate)

    df_failure_rates = pd.DataFrame(failure_rates)

    # barplot of trend of failure rates over time
    fig = px.bar(df_failure_rates, x='year', y='Failure Rate per km', title='Failure Rate Trend Over Time')

    return df_failure_rates, fig


@st.cache_data
def overall_failure_rate_modelling(df_failure_rates):
    """ Modelling failure rate per km for a Monte Carlo simulation - Linear Projection with Normally Distributed Residual Uncertainty """

    years = df_failure_rates['year'].tolist()
    rates = df_failure_rates['Failure Rate per km'].tolist()
    slope, intercept, r_value, p_value, std_err = linregress(years, rates)

    st.write("**Linear regression results for failure rate trend:**")
    st.write(f"Slope: {slope:.4f}, Intercept: {intercept:.2f}, R-squared: {r_value**2:.4f}, P-value: {p_value:.4f}, Std Err: {std_err:.4f}")

    fitted_values = [slope * y + intercept for y in years]
    residuals = [r - f for r, f in zip(rates, fitted_values)]
    sigma = np.std(residuals, ddof=2)

    st.write(f"Estimated standard deviation of Normally distributed residuals (σ): {sigma:.4f}")

    # failure rate trend and projection chart
    # Projection horizon (e.g., next 5 years)
    forecast_midpoint = 2030

    # Projected rate
    mu = slope * forecast_midpoint + intercept

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(years, rates, label='Historical data', alpha=0.6)
    ax.plot(years, fitted_values, 'r-', label=f'Trend: y={slope:.3f}x+{intercept:.3f}')
    ax.plot(forecast_midpoint, mu, 'go', label=f'Projected rate - μ: {mu:.3f} with CI σ: {sigma:.3f}')
    # create a vertical confidence interval for the projection
    ax.set_xlabel('Year')
    ax.set_ylabel('Failure Rate (per km/year)')
    ax.legend()
    ax.set_title('Failure Rate Trend and Projection (Example: to 2030)')

    # scale down font sizes in plot
    ax.title.set_fontsize(6)
    ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_fontsize(6)
    ax.legend(fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)
    for line in ax.get_lines():
        line.set_markersize(3)

    fig2 = model_distribution_fits(residuals)

    return fig, fig2, residuals


@st.cache_data
def asset_failure_rates(df, total_asset_length):
    """ Calculate failure rates due to asset deterioration and create barplot of trend of failure rates over time """

    failure_rates = {
        'year': [],
        'Failure Rate per km': []
    }

    asset_failure_causes = ['Conductor damage mid span', 'Insulator failure reported on cross-arm', 'UG cable insulation breakdown', 'Cable joint failure suspected']
    unknown_causes = ['HV trip – limited historical detail', 'HV trip – cause unclear, no defects found', 'Unknown cause – reset successful']

    u_frac = 0.2
    for y in sorted(list(df['year'].unique())):
        df_year = df[df['year'] == y]
        df_year_asset_failures = df_year[df_year['fault_description'].isin(asset_failure_causes)]
        df_year_unknown_failures = df_year[df_year['fault_description'].isin(unknown_causes)]
        num_failures = len(df_year_asset_failures) + u_frac * len(df_year_unknown_failures)
        failure_rate = num_failures / total_asset_length
        failure_rates['year'].append(y)
        failure_rates['Failure Rate per km'].append(failure_rate)

    df_failure_rates = pd.DataFrame(failure_rates)

    # barplot of trend of failure rates over time
    fig = px.bar(df_failure_rates, x='year', y='Failure Rate per km', title='Failure Rate Trend Over Time')

    return df_failure_rates, fig


@st.cache_data
def asset_failures_recent_trend(df_failure_rates_asset):
    """ Visual and statistical check of recent failure rates to assess if it is "stationary enough" """

    recent_period = 10

    # Check if recent period is "stationary enough"
    recent = df_failure_rates_asset['Failure Rate per km'].tail(recent_period)

    # 1. Visual check
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(recent, 'o-')
    ax.axhline(np.mean(recent), color='r', linestyle='--')
    
    # 2. Statistical check
    x = np.arange(len(recent))
    slope, _, _, p_val, _ = linregress(x, recent)

    ax.set_title(f'Recent Asset Deterioration Failure Rates - stable\nLin Regression: p-value for trend ({p_val:.3f}) > 0.1 and abs(slope) ({abs(slope):.3f}) < 0.1')
    ax.set_xlabel('Year')
    ax.set_ylabel('Failure Rate per km')

    # scale down font sizes in plot
    ax.title.set_fontsize(6)
    ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_fontsize(6)
    ax.legend(fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)
    for line in ax.get_lines():
        line.set_markersize(3)

    fig2 = model_distribution_fits(recent)
    return fig, fig2


@st.cache_data
def gamma_fit(df_failure_rates_asset):
    """ Fit a Gamma distribution to the recent failure rates due to asset deterioration for use in Monte Carlo simulation """

    recent_years = 10
    recent_rates = df_failure_rates_asset['Failure Rate per km'].tail(recent_years)

    # Fit Gamma distribution (good for positive skewed data)
    shape, loc, scale = gamma.fit(recent_rates, floc=0)  # floc=0 forces location=0

    st.write(f"**Gamma distribution parameters:** shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")
    st.write(f"Mean: {shape*scale:.3f}, Std: {np.sqrt(shape)*scale:.3f}")

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.hist(recent_rates, bins=6, density=True, alpha=0.6, label='Recent Data')
    x = np.linspace(0, max(recent_rates)*1.5, 100)
    ax.plot(x, gamma.pdf(x, shape, loc, scale), 'r-', label='Gamma fit')
    ax.legend()
    ax.set_title('Deterioration Failure Rate Distribution')

    # scale down font sizes in plot
    ax.title.set_fontsize(6)
    ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_fontsize(6)
    ax.legend(fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)
    for line in ax.get_lines():
        line.set_markersize(3)
    return fig

############################ Repair Time Modelling #########################
@st.cache_data
def repair_time_histograms(df):
    """ Histograms of repair times for all fault types and asset deterioration related faults in plotly """

    df_asset_deterioration = df[df['fault_description'].isin(['Conductor damage mid span', 'Insulator failure reported on cross-arm', 'UG cable insulation breakdown', 'Cable joint failure suspected'])]
    
    fig1 = px.histogram(df, x='fault_duration_minutes', nbins=50, title='Distribution of Repair Times for All Fault Types')
    fig2 = px.histogram(df_asset_deterioration, x='fault_duration_minutes', nbins=50, title='Distribution of Repair Times for Asset Deterioration Related Faults')
    
    return fig1, fig2
