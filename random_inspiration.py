import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, pearsonr, f_oneway
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from getpass import getpass
import plotly.express as px
import logging
import json
import os
import sys

def password_protect():
    """Prompt the user for a password to protect the script."""
    password = getpass("Enter the password to access the data analysis script: ")
    if password != config['password']:
        print("Incorrect password. Access denied.")
        exit()

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def validate_data(data):
    """Validate the structure and content of the input data."""
    required_columns = ['date', 'value', 'group']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    if data.isnull().values.any():
        raise ValueError("Data contains missing values.")
    try:
        pd.to_datetime(data['date'])
    except ValueError:
        raise ValueError("Invalid date format in 'date' column.")

def clean_data(data):
    """Clean the data by handling missing values and converting data types."""
    data = data.dropna()
    data['date'] = pd.to_datetime(data['date'])
    return data

def preprocess_data(data):
    """Preprocess the data for analysis."""
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    return data

def save_clean_data(data, file_path):
    """Save cleaned and preprocessed data to a CSV file."""
    data.to_csv(file_path, index=False)

def detect_outliers(data):
    """Detect outliers in the data using the IQR method."""
    Q1 = data['value'].quantile(0.25)
    Q3 = data['value'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data['value'] < (Q1 - 1.5 * IQR)) | (data['value'] > (Q3 + 1.5 * IQR))]
    return outliers

def handle_outliers(data):
    """Handle outliers by capping them to the whiskers."""
    Q1 = data['value'].quantile(0.25)
    Q3 = data['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data['value'] = np.where(data['value'] < lower_bound, lower_bound, data['value'])
    data['value'] = np.where(data['value'] > upper_bound, upper_bound, data['value'])
    return data

def exploratory_data_analysis(data):
    """Perform exploratory data analysis with summary statistics and visualizations."""
    summary = data.describe()
    correlation_matrix = data.corr()

    # Correlation Matrix Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Value Distribution Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data['value'], bins=30, kde=True)
    plt.title('Value Distribution')
    plt.show()

    # Boxplot of Values by Group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='value', data=data)
    plt.title('Boxplot of Values by Group')
    plt.show()

    # Lineplot of Value Over Time
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='value', data=data)
    plt.title('Value Over Time')
    plt.show()

    # Modern interactive graph with Plotly
    fig = px.line(data, x='date', y='value', title='Interactive Value Over Time')
    fig.show()

    return summary, correlation_matrix

def statistical_analysis(data):
    """Perform statistical analysis to compare groups and find correlations."""
    # T-test for comparing two groups
    group1 = data[data['group'] == 'A']['value']
    group2 = data[data['group'] == 'B']['value']
    t_stat, p_value = ttest_ind(group1, group2)

    # Chi-squared test for categorical data (if applicable)
    chi2, chi2_p_value = None, None
    if 'category' in data.columns:
        contingency_table = pd.crosstab(data['group'], data['category'])
        chi2, chi2_p_value, _, _ = chi2_contingency(contingency_table)

    # Pearson correlation for numerical data
    pearson_corr, pearson_p_value = None, None
    if 'another_value' in data.columns:
        pearson_corr, pearson_p_value = pearsonr(data['value'], data['another_value'])

    # ANOVA test for comparing more than two groups
    anova_f_stat, anova_p_value = None, None
    if data['group'].nunique() > 2:
        anova_groups = [data['value'][data['group'] == group] for group in data['group'].unique()]
        anova_f_stat, anova_p_value = f_oneway(*anova_groups)

    return t_stat, p_value, chi2, chi2_p_value, pearson_corr, pearson_p_value, anova_f_stat, anova_p_value

def train_model(data):
    """Train and evaluate multiple machine learning models to predict values."""
    X = data[['year', 'month', 'day']]
    y = data['value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'mse': mse, 'r2': r2}

    # Hyperparameter tuning for Ridge regression
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results['Ridge_GridSearch'] = {'model': best_model, 'mse': mse, 'r2': r2}

    return results

def save_results(summary, correlation_matrix, t_stat, p_value, chi2, chi2_p_value, pearson_corr, pearson_p_value, anova_f_stat, anova_p_value, model_results, output_path):
    """Save the analysis results to a text file."""
    with open(output_path, 'w') as f:
        f.write('Summary Statistics:\n')
        f.write(summary.to_string())
        f.write('\n\nCorrelation Matrix:\n')
        f.write(correlation_matrix.to_string())
        f.write('\n\nT-Test Results:\n')
        f.write(f'T-Statistic: {t_stat}\n')
        f.write(f'P-Value: {p_value}\n')
        if chi2 is not None:
            f.write('\n\nChi-Squared Test Results:\n')
            f.write(f'Chi-Squared: {chi2}\n')
            f.write(f'P-Value: {chi2_p_value}\n')
        if pearson_corr is not None:
            f.write('\n\nPearson Correlation Results:\n')
            f.write(f'Correlation Coefficient: {pearson_corr}\n')
            f.write(f'P-Value: {pearson_p_value}\n')
        if anova_f_stat is not None:
            f.write('\n\nANOVA Test Results:\n')
            f.write(f'F-Statistic: {anova_f_stat}\n')
            f.write(f'P-Value: {anova_p_value}\n')
        f.write('\n\nMachine Learning Model Evaluation:\n')
        for name, result in model_results.items():
            f.write(f'\n{name}:\n')
            f.write(f'Mean Squared Error: {result["mse"]}\n')
            f.write(f'R-Squared: {result["r2"]}\n')

def generate_report(summary, correlation_matrix, t_stat, p_value, chi2, chi2_p_value, pearson_corr, pearson_p_value, anova_f_stat, anova_p_value, model_results, output_path):
    """Generate a detailed HTML report with all findings and visualizations."""
    from jinja2 import Environment

, FileSystemLoader

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('report_template.html')

    report_html = template.render(
        summary=summary.to_html(),
        correlation_matrix=correlation_matrix.to_html(),
        t_stat=t_stat,
        p_value=p_value,
        chi2=chi2,
        chi2_p_value=chi2_p_value,
        pearson_corr=pearson_corr,
        pearson_p_value=pearson_p_value,
        anova_f_stat=anova_f_stat,
        anova_p_value=anova_p_value,
        model_results=model_results
    )

    with open(output_path, 'w') as f:
        f.write(report_html)

if __name__ == "__main__":
    # Load configuration
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Set up logging
    logging.basicConfig(filename='data_analysis.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    password_protect()

    try:
        logging.info("Loading data...")
        data = load_data(config['data_file'])
        validate_data(data)

        logging.info("Cleaning data...")
        data = clean_data(data)

        logging.info("Preprocessing data...")
        data = preprocess_data(data)
        save_clean_data(data, config['clean_data_file'])

        logging.info("Detecting outliers...")
        outliers = detect_outliers(data)
        logging.info(f"Detected outliers:\n{outliers}")

        logging.info("Handling outliers...")
        data = handle_outliers(data)

        logging.info("Performing exploratory data analysis...")
        summary, correlation_matrix = exploratory_data_analysis(data)

        logging.info("Performing statistical analysis...")
        t_stat, p_value, chi2, chi2_p_value, pearson_corr, pearson_p_value, anova_f_stat, anova_p_value = statistical_analysis(data)

        logging.info("Training machine learning models...")
        model_results = train_model(data)

        logging.info("Saving results...")
        save_results(summary, correlation_matrix, t_stat, p_value, chi2, chi2_p_value, pearson_corr, pearson_p_value, anova_f_stat, anova_p_value, model_results, config['results_file'])

        logging.info("Generating report...")
        generate_report(summary, correlation_matrix, t_stat, p_value, chi2, chi2_p_value, pearson_corr, pearson_p_value, anova_f_stat, anova_p_value, model_results, config['report_file'])

        logging.info("Data analysis completed successfully.")

    except Exception as e:
        logging.error("An error occurred during data analysis.", exc_info=True)
