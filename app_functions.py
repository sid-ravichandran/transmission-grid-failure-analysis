import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, probplot, norm

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


############################ Failure Rate Analysis #########################
@st.cache_data
def model_normal_distribution_fit(input_data):
    """ Fit Normal distribution to input fault rate (residual) data for use in Monte Carlo simulation """

    fig, ax = plt.subplots(figsize=(3, 3))
    _, (slope, intercept, r) = probplot(input_data, sparams=(1,), dist='norm', plot=ax)
    ax.set_title(f"Q-Q for Normal\nR-squared: {r**2:.4f}")

    plt.tight_layout()
    # scale down font sizes in plot and dot size in Q-Q plot
    for line in ax.get_lines():
        line.set_markersize(3)
    ax.title.set_fontsize(6)
    ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_fontsize(6)
    ax.legend(fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)

    return fig


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
def failure_rate_modelling(df_failure_rates, cause='all'):
    """ Modelling failure rate per km for a Monte Carlo simulation - Linear Projection with Normally Distributed Residual Uncertainty """

    years = df_failure_rates['year'].tolist()
    rates = df_failure_rates['Failure Rate per km'].tolist()
    slope, intercept, r_value, p_value, std_err = linregress(years, rates)

    st.write("**Linear regression results for failure rate trend:**")
    st.write(f"Slope: {slope:.4f}, Intercept: {intercept:.2f}, R-squared: {r_value**2:.4f}")

    if cause == 'all':
        st.write(f"The overall failure rate/km is increasing by approximately {slope:.4f}/km per year. The high R-squared value indicates that the linear model explains almost all of the variability in the failure rates.")
    elif cause == 'asset':
        st.write(f"The asset deterioration related failure rate/km is increasing by approximately {slope:.4f}/km per year. The lower R-squared value indicates that the linear model does not capture all of variability in the failure rates.")

    fitted_values = [slope * y + intercept for y in years]
    residuals = [r - f for r, f in zip(rates, fitted_values)]
    sigma = np.std(residuals, ddof=2)

    st.write(f"Estimated standard deviation of Normally distributed residuals (σ): {sigma:.4f}")

    # Histogram distribution of residuals
    fig0, ax0 = plt.subplots(figsize=(3, 3))
    sns.histplot(residuals, kde=True, ax=ax0, bins=10)
    ax0.set_title('Distribution of Residuals from Linear Fit')
    ax0.set_xlabel('Residual (Actual - Fitted Failure Rate)')
    ax0.set_ylabel('Frequency')
    # scale down font sizes in plot
    ax0.title.set_fontsize(6)
    ax0.xaxis.label.set_fontsize(6)
    ax0.yaxis.label.set_fontsize(6)
    ax0.legend(fontsize=5)
    ax0.tick_params(axis='both', which='major', labelsize=5)

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

    fig2 = model_normal_distribution_fit(residuals)

    return fig0, fig, fig2, residuals


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


############################ Repair Time Modelling #########################
@st.cache_data
def model_lognormal_distribution_fit(input_data):
    """ Fit Lognormal distribution to input fault rate (residual) data for use in Monte Carlo simulation """

    fig, ax = plt.subplots(figsize=(3, 3))
    shape, loc, scale = stats.lognorm.fit(input_data)
    st.write(f"**LogNormal distribution parameters:** shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")
    
    _, (slope, intercept, r) = probplot(input_data, sparams=(shape, loc, scale), dist='lognorm', plot=ax)
    ax.set_title(f"Q-Q for Lognormal\nR-squared: {r**2:.4f}")

    plt.tight_layout()
    # scale down font sizes in plot and dot size in Q-Q plot
    for line in ax.get_lines():
        line.set_markersize(3)
    ax.title.set_fontsize(6)
    ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_fontsize(6)
    ax.legend(fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)

    return fig


@st.cache_data
def repair_time_histograms(df):
    """ Histograms of repair times for all fault types and asset deterioration related faults in plotly """

    df_asset_deterioration = df[df['fault_description'].isin(['Conductor damage mid span', 'Insulator failure reported on cross-arm', 'UG cable insulation breakdown', 'Cable joint failure suspected'])]
    
    fig1 = px.histogram(df, x='fault_duration_minutes_per_km', nbins=50, title='Distribution of Repair Times for All Fault Types', labels={'fault_duration_minutes_per_km': 'Customer Time Lost per km (minutes/km)'})
    fig2 = px.histogram(df_asset_deterioration, x='fault_duration_minutes_per_km', nbins=50, title='Distribution of Repair Times for Asset Deterioration Related Faults', labels={'fault_duration_minutes_per_km': 'Customer Time Lost per km (minutes/km)'})
    
    # Repair duration per km by year
    df_yearly = df.groupby('year')['fault_duration_minutes_per_km'].sum().reset_index()
    fig3 = px.bar(df_yearly, x='year', y='fault_duration_minutes_per_km', 
                   title='Total Repair Duration per km by Year',
                   labels={'fault_duration_minutes_per_km': 'Total Duration (min/km)', 'year': 'Year'})
    fig3.update_xaxes(tickangle=-45)
    
    return fig1, fig2, fig3


@st.cache_data
def repair_rate_modelling(df):
    """ Modelling repair time per km for a Monte Carlo simulation - Linear Projection with Normally Distributed Residual Uncertainty """

    df_yearly = df.groupby('year')['fault_duration_minutes_per_km'].sum().reset_index()
    years = df_yearly['year'].tolist()
    rates = df_yearly['fault_duration_minutes_per_km'].tolist()
    slope, intercept, r_value, p_value, std_err = linregress(years, rates)

    st.write("**Linear regression results for repair time trend:**")
    st.write(f"Slope: {slope:.4f}, Intercept: {intercept:.2f}, R-squared: {r_value**2:.4f}")

    st.write(f"The overall repair time/km is increasing by approximately {slope:.4f} min/km per year. The R-squared value indicates that the linear model explains almost all of the variability in the repair times.")

    fitted_values = [slope * y + intercept for y in years]
    residuals = [r - f for r, f in zip(rates, fitted_values)]
    sigma = np.std(residuals, ddof=2)

    st.write(f"Estimated standard deviation of the residuals (σ): {sigma:.4f}")

    # Histogram distribution of residuals
    fig0, ax0 = plt.subplots(figsize=(3, 3))
    sns.histplot(residuals, kde=True, ax=ax0, bins=10)
    ax0.set_title('Distribution of Residuals from Linear Fit')
    ax0.set_xlabel('Residual (Actual - Fitted Failure Rate)')
    ax0.set_ylabel('Frequency')
    # scale down font sizes in plot
    ax0.title.set_fontsize(6)
    ax0.xaxis.label.set_fontsize(6)
    ax0.yaxis.label.set_fontsize(6)
    ax0.legend(fontsize=5)
    ax0.tick_params(axis='both', which='major', labelsize=5)

    fig2 = model_normal_distribution_fit(residuals)

    return fig0, fig2, residuals
