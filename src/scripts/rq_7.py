#!/usr/bin/env python
# coding: utf-8

# # Research question 7

# ### How do movie endings vary by film length ? This question would explore whether movies of different lengths (e.g., short vs. feature-length) tend to have different types of endings.
# 
# This notebook presents initial observations and is not intended to represent the final conclusions.

# ##### Importations

# In[1]:


import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kstest, mannwhitneyu, levene, ttest_ind, normaltest
from sklearn.utils import resample
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import normaltest, levene, f_oneway, kruskal
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# path
DATA_FOLDER = 'src/data/'
MOVIE_DATASET = DATA_FOLDER + 'movies_dataset_final.tsv'

# Dataset loading
movies = pd.read_csv(MOVIE_DATASET, sep='\t')


# In[3]:


def rebalance_data(movies_cleaned, threshold=40):
    # Split the movies into two groups: short and feature-length
    short_movies = movies_cleaned[movies_cleaned['duration_category'] == 'Short']
    feature_movies = movies_cleaned[movies_cleaned['duration_category'] == 'Feature-length']
    
    # Under-sampling the longer movies group to balance the sizes
    if len(short_movies) < len(feature_movies):
        feature_movies_resampled = resample(feature_movies, 
                                            replace=False,  # Without replacement
                                            n_samples=len(short_movies),  # Size equal to the minority group
                                            random_state=42)
        movies_balanced = pd.concat([short_movies, feature_movies_resampled])
    else:
        short_movies_resampled = resample(short_movies, 
                                          replace=False,  # Without replacement
                                          n_samples=len(feature_movies),  # Size equal to the minority group
                                          random_state=42)
        movies_balanced = pd.concat([feature_movies, short_movies_resampled])
    
    return movies_balanced


# ### Test for two categories

# These statistical analyses were conducted to compare movie scores based on film length, differentiated by a specified threshold. The tests used (such as normality tests, Levene's test for equality of variances, T-tests, and Mann-Whitney U tests) ensure that the assumptions of the selected statistical methods are met. Normality tests verify if the score distributions follow a normal distribution, while Levene's test checks for homogeneity of variances between the two categories (short and feature-length). Depending on these results, either a T-test (assuming equal variances) or Welch's T-test (if variances are unequal) can be used, or the non-parametric Mann-Whitney U test is applied if necessary. This systematic approach guarantees accurate statistical comparison while considering the underlying data characteristics.

# In[4]:


def analyze_movie_endings_by_threshold(movies, threshold=40, verbose=True):
    # Check the required columns
    if 'Movie runtime' not in movies.columns or 'Score' not in movies.columns:
        raise ValueError("The DataFrame must contain the columns 'Movie runtime' and 'Score'")
    
    # Clean the data
    movies['Movie runtime'] = pd.to_numeric(movies['Movie runtime'], errors='coerce')
    movies_cleaned = movies.dropna(subset=['Movie runtime', 'Score'])
    movies_cleaned = movies_cleaned[movies_cleaned['Movie runtime'] > 0]
    
    # Categorize by threshold
    movies_cleaned['duration_category'] = movies_cleaned['Movie runtime'].apply(
        lambda x: 'Short' if x < threshold else 'Feature-length'
    )
    
    # Rebalance the data
    movies_cleaned = rebalance_data(movies_cleaned, threshold)
    
    # Separate the scores by category
    short_scores = movies_cleaned[movies_cleaned['duration_category'] == 'Short']['Score']
    feature_scores = movies_cleaned[movies_cleaned['duration_category'] == 'Feature-length']['Score']
    
    # Descriptive statistics
    group_stats = movies_cleaned.groupby('duration_category')['Score'].agg(['mean', 'median', 'std', 'count'])
    if verbose:
        print("Descriptive Statistics by Duration Category:")
        print(group_stats.to_string(index=True, float_format="{:.2f}".format))
    
    # Normality test (D'Agostino and Pearson)
    normality_short = normaltest(short_scores)
    normality_feature = normaltest(feature_scores)
    if verbose:
        print(f"\nNormality Test (D'Agostino and Pearson):")
        print(f"Short Movies: Stat={normality_short.statistic:.4f}, p-value={normality_short.pvalue:.4f}")
        print(f"Feature-length Movies: Stat={normality_feature.statistic:.4f}, p-value={normality_feature.pvalue:.4f}")
    
    # Interpretation of normality results
    if verbose:
        if normality_short.pvalue > 0.05:
            print("Short movies distribution follows a normal distribution.")
        else:
            print("Short movies distribution does not follow a normal distribution.")
        
        if normality_feature.pvalue > 0.05:
            print("Feature-length movies distribution follows a normal distribution.")
        else:
            print("Feature-length movies distribution does not follow a normal distribution.")
    
    # Variance homogeneity test (Levene)
    levene_stat, levene_p = levene(short_scores, feature_scores)
    if verbose:
        print(f"\nLevene's Test for Equality of Variances: Statistic={levene_stat:.4f}, p-value={levene_p:.4f}")
    
    # Interpretation of Levene's test
    if verbose:
        if levene_p > 0.05:
            print("Variances between short and feature-length movies are equal.")
        else:
            print("Variances between short and feature-length movies are not equal.")
    
    # Choosing the statistical test
    if normality_short.pvalue > 0.05 and normality_feature.pvalue > 0.05:
        if levene_p > 0.05:
            # T-test if normality and homogeneity of variances
            test_stat, test_p = ttest_ind(short_scores, feature_scores)
            test_name = "T-Test (Independent Samples)"
        else:
            # Welch's T-test if variances are unequal
            test_stat, test_p = ttest_ind(short_scores, feature_scores, equal_var=False)
            test_name = "Welch's T-Test"
    else:
        # Mann-Whitney U test if non-parametric
        test_stat, test_p = mannwhitneyu(short_scores, feature_scores)
        test_name = "Mann-Whitney U Test"
    
    if verbose:
        print(f"\n{test_name} Results:")
        print(f"Statistic: {test_stat:.4f}, p-value: {test_p:.4f}")
        if test_p < 0.05:
            print(f"The difference between short and feature-length movies is statistically significant.")
            
            # Comparing the means if significant
            short_mean = short_scores.mean()
            feature_mean = feature_scores.mean()
            print(f"Mean Score for Short Movies: {short_mean:.2f}")
            print(f"Mean Score for Feature-length Movies: {feature_mean:.2f}")
            
            if feature_mean > short_mean:
                print(f"The mean score increases for feature-length movies compared to short movies.")
            else:
                print(f"The mean score does not increase for feature-length movies compared to short movies.")
        else:
            print(f"The difference between short and feature-length movies is not statistically significant.")
    
    # Visualization
    if verbose:
        plt.figure(figsize=(5, 3))
        sns.boxplot(data=movies_cleaned, x='duration_category', y='Score')
        plt.title(f'Distribution of Scores by Film Length (Threshold = {threshold} minutes)')
        plt.show()
    
    # Return the results
    return {
        "group_stats": group_stats,
        "normality": {
            "short": {"statistic": normality_short.statistic, "p_value": normality_short.pvalue},
            "feature": {"statistic": normality_feature.statistic, "p_value": normality_feature.pvalue},
        },
        "levene": {"statistic": levene_stat, "p_value": levene_p},
        "test": {
            "name": test_name,
            "statistic": test_stat,
            "p_value": test_p
        }
    }


# In[5]:


movies = pd.read_csv(MOVIE_DATASET, sep='\t')

thresholds = range(10, 241, 10)

previous_short_mean = None
previous_feature_mean = None

for threshold in thresholds:
    results = analyze_movie_endings_by_threshold(movies, threshold=threshold, verbose=False)
    p_value = results['test']['p_value']
    
    if p_value < 0.05:
        short_mean = results['group_stats'].loc['Short', 'mean']
        feature_mean = results['group_stats'].loc['Feature-length', 'mean']
        
        print(f"Threshold: {threshold}, p-value: {p_value:.4f} - Significant")
        print(f"Mean Score for Short Movies: {short_mean:.2f}")
        print(f"Mean Score for Feature-length Movies: {feature_mean:.2f}")
        
        # Compare with previous threshold to see if the means are increasing or decreasing
        if previous_short_mean is not None and previous_feature_mean is not None:
            short_diff = short_mean - previous_short_mean
            feature_diff = feature_mean - previous_feature_mean
            
            # Affichage de l'évolution
            if short_diff > 0:
                print(f"Short films mean increased by {short_diff:.2f} compared to previous threshold")
            elif short_diff < 0:
                print(f"Short films mean decreased by {abs(short_diff):.2f} compared to previous threshold")
            else:
                print(f"Short films mean remained the same compared to previous threshold")
            
            if feature_diff > 0:
                print(f"Feature-length films mean increased by {feature_diff:.2f} compared to previous threshold")
            elif feature_diff < 0:
                print(f"Feature-length films mean decreased by {abs(feature_diff):.2f} compared to previous threshold")
            else:
                print(f"Feature-length films mean remained the same compared to previous threshold")
        
        # Update previous means for the next comparison
        previous_short_mean = short_mean
        previous_feature_mean = feature_mean
        
        print(f"Difference in means: {feature_mean - short_mean:.2f}\n")


new_thresholds = [100, 110, 130, 140, 210, 220]
for threshold in new_thresholds:
    results = analyze_movie_endings_by_threshold(movies, threshold=threshold, verbose=True)
    p_value = results['test']['p_value']


# In[ ]:


# Variables to store all means and thresholds
all_thresholds = []
all_short_means = []
all_feature_means = []
significant_indices = []  # To track which thresholds are significant

# Loop over thresholds
thresholds = range(10, 241, 10)  # Thresholds from 10 to 240 minutes
for i, threshold in enumerate(thresholds):
    results = analyze_movie_endings_by_threshold(movies, threshold=threshold, verbose=False)
    p_value = results['test']['p_value']
    
    # Extract mean values
    short_mean = results['group_stats'].loc['Short', 'mean']
    feature_mean = results['group_stats'].loc['Feature-length', 'mean']
    
    # Append means to lists
    all_thresholds.append(threshold)
    all_short_means.append(short_mean)
    all_feature_means.append(feature_mean)
    
    # Mark index as significant if p-value is below 0.05
    if p_value < 0.05:
        significant_indices.append(i)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot all means for short and feature-length movies
plt.plot(all_thresholds, all_short_means, label='Short Movies Mean', linestyle='-', color='#f2e4d5', linewidth=3.5)
plt.plot(all_thresholds, all_feature_means, label='Feature-length Movies Mean', linestyle='-', color='#d1b8a1', linewidth=3.5)

# Highlight significant thresholds with different marker colors
for idx in significant_indices:
    plt.scatter(all_thresholds[idx], all_short_means[idx], color='brown', s=100, label='Significant Threshold' if idx == significant_indices[0] else "")
    plt.scatter(all_thresholds[idx], all_feature_means[idx], color='brown', s=100, label='Significant Threshold' if idx == significant_indices[0] else "")

# Add labels, title, and legend
plt.xlabel('Threshold (Minutes)', fontsize=12)
plt.ylabel('Mean Score', fontsize=12)
plt.title('Mean Scores by Movie Length Threshold (All Thresholds)', fontsize=14)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show the plot
plt.show()


# In[7]:


# Data cleaning step (done only once)
movies['Movie runtime'] = pd.to_numeric(movies['Movie runtime'], errors='coerce')
movies_cleaned = movies.dropna(subset=['Movie runtime', 'Score'])
movies_cleaned = movies_cleaned[movies_cleaned['Movie runtime'] > 0]

# Function to generate an interactive boxplot with a dropdown menu for the threshold
def create_interactive_boxplot(movies, thresholds, filename):
    fig = go.Figure()

    # Initialize an empty list to store traces
    traces = []

    # Add traces for each threshold, but initially set them to 'legendonly' (hidden)
    for idx, threshold in enumerate(thresholds):
        # Filter the movies based on the threshold
        short_movies = movies[movies['Movie runtime'] < threshold]
        feature_movies = movies[movies['Movie runtime'] >= threshold]

        # Create traces for short films and feature-length films
        traces.append(go.Box(
            y=short_movies['Score'],
            name=f'Short (Threshold {threshold} min)',
            boxmean='sd',
            marker_color='#d6bfa6',  # Color for short films
            visible=True if threshold == thresholds[0] else False,  # Show only the first threshold by default
            legendgroup=f"threshold_{threshold}"  # Group by threshold for easy control
        ))

        traces.append(go.Box(
            y=feature_movies['Score'],
            name=f'Feature-length (Threshold {threshold} min)',
            boxmean='sd',
            marker_color='#b59e89',  # Color for feature-length films
            visible=True if threshold == thresholds[0] else False,  # Show only the first threshold by default
            legendgroup=f"threshold_{threshold}"  # Group by threshold for easy control
        ))

    # Add buttons to the dropdown menu
    buttons = [
        {
            'label': f'Threshold {threshold} min',
            'method': 'update',
            'args': [
                # Set visibility of traces based on the selected threshold
                {
                    'visible': [
                        # Only show the traces for the selected threshold
                        True if f"threshold_{threshold}" in trace['legendgroup'] else False
                        for trace in traces
                    ]
                },
                {'title': f'Distribution of Scores by Movie Duration (Threshold {threshold} min)'}
            ],
        }
        for threshold in thresholds
    ]

    # Add traces to the figure
    fig.add_traces(traces)

    # Update the layout with a dropdown menu at the top-right and remove modebar (zoom, pan)
    fig.update_layout(
        title=f'Distribution of Scores by Movie Duration',
        xaxis_title='Movie Categories',
        yaxis_title='Scores',
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1,  # Position it at the far right
            'xanchor': 'right',  # Anchor the menu to the right
            'y': 1.1,  # Position it slightly above the plot
            'yanchor': 'top'  # Anchor it to the top
        }],
        # Disable modebar (zoom, pan, etc.)
        modebar=dict(
            remove=['zoom', 'pan', 'zoomIn', 'zoomOut', 'resetScale2d', 'lasso2d', 'select2d', 'zoom3d', 'turntable', 'orbit', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'resetViewMapbox', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleHover', 'resetView', 'zoomInGeo', 'zoomOutGeo']
        )
    )

    fig.write_html(filename)
    # Show the plot with plotly.io
    pio.show(fig)

# Call the function with the desired thresholds
create_interactive_boxplot(movies_cleaned, thresholds=[30, 50, 80, 90, 110, 120, 150, 200, 210], filename='boxplot.html')


# ### Test for three categories

# These tests were conducted to assess whether movie scores differ significantly across three categories based on runtime: Short, Feature-length, and Long films. By performing normality tests (e.g., D'Agostino and Pearson's test), we ensure the data distribution aligns with assumptions for parametric tests. Levene's test for homogeneity of variances determines if the variability of scores is consistent across categories. Depending on these results, we use either ANOVA (for normal distributions and equal variances), Welch's ANOVA (for unequal variances), or the Kruskal-Wallis test (for non-normal distributions). This process allows for accurate statistical analysis of movie scores while accounting for differences in film length.

# In[6]:


import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


def analyze_movie_endings_three_categories(movies, thresholds=(40, 100), print_output=True):
    
    # Check necessary columns
    if 'Movie runtime' not in movies.columns or 'Score' not in movies.columns:
        raise ValueError("The DataFrame must contain the columns 'Movie runtime' and 'Score'")
    
    # Clean the data
    movies['Movie runtime'] = pd.to_numeric(movies['Movie runtime'], errors='coerce')
    movies_cleaned = movies.dropna(subset=['Movie runtime', 'Score'])
    movies_cleaned = movies_cleaned[movies_cleaned['Movie runtime'] > 0]
    
    # Categorize by thresholds
    low, high = thresholds
    movies_cleaned['duration_category'] = movies_cleaned['Movie runtime'].apply(
        lambda x: 'Short' if x < low else ('Feature-length' if x < high else 'Long')
    )
    
    # Downsampling to balance categories
    min_size = movies_cleaned['duration_category'].value_counts().min()
    downsampled_movies = (
        movies_cleaned.groupby('duration_category')
        .apply(lambda x: x.sample(n=min_size, random_state=42))
        .reset_index(drop=True)
    )
    
    if print_output:
        print(f"Downsampled categories to {min_size} movies each.")
    
    # Descriptive statistics
    group_stats = downsampled_movies.groupby('duration_category')['Score'].agg(['mean', 'median', 'std', 'count'])
    if print_output:
        print("\nDescriptive Statistics by Duration Category (Downsampled):")
        print(group_stats.to_string(index=True, float_format="{:.2f}".format))
    
    # Normality tests for each category
    normality_results = {}
    for category in ['Short', 'Feature-length', 'Long']:
        scores = downsampled_movies[downsampled_movies['duration_category'] == category]['Score']
        normality_results[category] = normaltest(scores)
        if print_output:
            print(f"\nNormality Test for {category} Movies: Stat={normality_results[category].statistic:.4f}, "
                  f"p-value={normality_results[category].pvalue:.4f}")
    
    # Levene's test for equality of variances
    short_scores = downsampled_movies[downsampled_movies['duration_category'] == 'Short']['Score']
    feature_scores = downsampled_movies[downsampled_movies['duration_category'] == 'Feature-length']['Score']
    long_scores = downsampled_movies[downsampled_movies['duration_category'] == 'Long']['Score']
    
    levene_stat, levene_p = levene(short_scores, feature_scores, long_scores)
    if print_output:
        print(f"\nLevene's Test for Equality of Variances: Statistic={levene_stat:.4f}, p-value={levene_p:.4f}")
    
    # Choosing the statistical test
    if all(normality_results[cat].pvalue > 0.05 for cat in ['Short', 'Feature-length', 'Long']):
        if levene_p > 0.05:
            # ANOVA if normality and equality of variances
            test_stat, test_p = f_oneway(short_scores, feature_scores, long_scores)
            test_name = "ANOVA (One-Way)"
        else:
            # Welch ANOVA if unequal variances
            test_stat, test_p = f_oneway(short_scores, feature_scores, long_scores)  # Welch-ANOVA requires statsmodels
            test_name = "Welch's ANOVA"
    else:
        # Kruskal-Wallis if non-parametric
        test_stat, test_p = kruskal(short_scores, feature_scores, long_scores)
        test_name = "Kruskal-Wallis Test"
    
    if print_output:
        print(f"\n{test_name} Results:")
        print(f"Statistic: {test_stat:.4f}, p-value: {test_p:.4f}")
    
    # Interpretation of the test
    if print_output:
        if test_p < 0.05:
            print("The differences between the three categories are statistically significant.")
        else:
            print("The differences between the three categories are not statistically significant.")
    
    # Visualization
    if print_output:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=downsampled_movies, x='duration_category', y='Score')
        plt.title(f'Distribution of Scores by Film Length Categories (Thresholds = {thresholds}, Downsampled)')
        plt.show()
    
    # Return results
    return {
        "group_stats": group_stats,
        "normality": {
            cat: {"statistic": normality_results[cat].statistic, "p_value": normality_results[cat].pvalue}
            for cat in ['Short', 'Feature-length', 'Long']
        },
        "levene": {"statistic": levene_stat, "p_value": levene_p},
        "test": {
            "name": test_name,
            "statistic": test_stat,
            "p_value": test_p
        }
    }


# In[7]:


movies_df = pd.read_csv(MOVIE_DATASET, sep='\t')

threshold_pairs = [
    (10, 60), (15, 70), (20, 80), (25, 90), (30, 100),
    (35, 110), (40, 120), (45, 130), (50, 140), (55, 150),
    (60, 160), (65, 170), (70, 180), (75, 190), (80, 200),
    (85, 210), (90, 220), (100, 130), (110, 150), (120, 160),
    (130, 170), (140, 180), (150, 190), (160, 200), (170, 210),
    (180, 220), (190, 210), (200, 220), (150, 200), (130, 180)
]

print(f"\n--- Analyzing with 3 categories: ---")
for thresholds in threshold_pairs:
    #print(f"\n--- Analyzing with thresholds: {thresholds} ---")
    result = analyze_movie_endings_three_categories(
        movies_df, thresholds=thresholds, print_output=False
    )

    p_value = result['test']['p_value']
    if p_value < 0.05:
        print(f"Threshold: {thresholds}, p-value: {p_value:.4f} - Significant")


# In[ ]:




