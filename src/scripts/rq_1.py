#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets, Output
from IPython.display import display
from scipy.stats import shapiro, levene, f_oneway, kruskal
import scikit_posthocs as sp
from scikit_posthocs import posthoc_dunn
import matplotlib.colors as mcolors

# Define constants
DATA_FOLDER = '../data/'
MOVIE_DATASET = DATA_FOLDER + 'movies_dataset_final.tsv'
copper_colorscale = [
    (255 / 255, 242 / 255, 230 / 255),
    (230 / 255, 169 / 255, 148 / 255),
    (202 / 255, 94 / 255, 91 / 255),
    (153 / 255, 51 / 255, 51 / 255),
    (102 / 255, 0 / 255, 0 / 255)
]
copper_interpolated = mcolors.LinearSegmentedColormap.from_list("copper", copper_colorscale)


# Function to load dataset
def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')

# Function to clean genres and filter based on frequency
def clean_and_filter_genres(movies, threshold=500):

    #Drop rows with no score
    movies['Movie genres'] = movies['Movie genres'].str.lower()
    genre_counts = movies['Movie genres'].str.split(', ').explode().value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Number of movies']
    genre_counts = genre_counts.sort_values(by='Number of movies', ascending=False)
    print("Number of genres in the dataset:", len(genre_counts))
    genre_counts = genre_counts[genre_counts['Number of movies'] > threshold]
    print("Number of genres we keep for our analysis:", len(genre_counts), "(Genres with more than", threshold, "movies)")
    # New dataframe 
    movies_genres = movies.copy()
    movies_genres = movies_genres.dropna(subset=['Movie genres'])
    movies_genres = movies_genres.dropna(subset=['Score'])
    movies_genres['Movie genres'] = movies_genres['Movie genres'].str.split(', ')
    movies_genres = movies_genres.explode('Movie genres')
    movies_genres = movies_genres.reset_index(drop=True)

    # Remove the rows of the movies_genres dataframe which genre is not in the genre_counts dataframe
    movies_genres = movies_genres[movies_genres['Movie genres'].isin(genre_counts['Genre'])]
    movies_genres = movies_genres.reset_index(drop=True)

    return movies_genres, genre_counts

# Function to visualize the number of movies per genre with a custom color scale
def plot_genre_counts(movies_genres):
    sample_sizes = movies_genres.groupby('Movie genres').size().reset_index(name='Count')
    sample_sizes = sample_sizes.sort_values(by='Count', ascending=False)

    # Interpolate colors for the genres
    num_genres = sample_sizes.shape[0]
    genre_colors = {
        genre: mcolors.rgb2hex(copper_interpolated(i / (num_genres - 1)))
        for i, genre in enumerate(sample_sizes['Movie genres'])
    }

    # Create the bar chart with the custom color scale
    fig = px.bar(
        sample_sizes,
        x='Movie genres',
        y='Count',
        color='Movie genres',
        title="Number of Movies per Genre",
        color_discrete_map=genre_colors
    )

    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Number of Movies",
        width=900,
        height=600
    )

    fig.show()

# Function to compute and visualize the average ending score per genre
def plot_average_ending_score(movies_genres):
    genre_scores = movies_genres.groupby('Movie genres').agg({'Score': 'mean', 'Movie genres': 'count'})
    genre_scores.columns = ['Mean score', 'Number of movies']
    copper_colormap = cm.get_cmap('copper', 256)  # 256 niveaux
    copper_colorscale = [
        [i / 255, f"rgb({int(255 * r)}, {int(255 * g)}, {int(255 * b)})"]
        for i, (r, g, b, _) in enumerate(copper_colormap(np.linspace(0, 1, 256)))
    ]
    genre_scores = genre_scores.reset_index()
    genre_scores = genre_scores.sort_values(by='Mean score', ascending=True)

    # Create bubble chart
    fig = px.scatter(
        genre_scores,
        x='Mean score',
        y='Movie genres',
        size='Number of movies',
        color='Mean score',
        color_continuous_scale=copper_colorscale,
        title='Distribution of genres by mean ending score',
        labels={'Mean_Score': 'Mean Score', 'Genre': 'Genre', 'Movie_Count': 'Number of Movies'}
    )

    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title="Mean score",
        yaxis_title="Genre",
        coloraxis_colorbar=dict(title="Mean Score"),
        height=600
    )

    fig.show()

# Function to create a plot for a specific genre and score distribution
def plot_genre_score_distribution(movies_genres, selected_genre):
    filtered_df = movies_genres[movies_genres['Movie genres'] == selected_genre]
    score_counts = filtered_df.groupby('Rounded Score').size().reset_index(name='Number of Movies')

    fig = px.bar(
        score_counts,
        x='Rounded Score',
        y='Number of Movies',
        hover_data={'Rounded Score': True, 'Number of Movies': True},
        title=f"Movie ending score distribution for the genre: {selected_genre}",
        labels={'Rounded Score': 'Score', 'Number of Movies': 'Number of Movies'},
        color_discrete_sequence=['#b87333'],
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="Number of Movies",
        height=400
    )

    return fig

# Function to handle the interactive dropdown for genre selection
def interactive_distribution_genre_plot(movies_genres):
    # Round scores to 2 decimals to aggregate close values
    movies_genres['Rounded Score'] = movies_genres['Score'].round(2)
    genres = movies_genres['Movie genres'].unique()

    genre_dropdown = widgets.Dropdown(
        options=genres,
        value=genres[0],
        description='Genre:',
    )

    output = Output()

    def update_plot(change):
        selected_genre = change['new']
        with output:
            output.clear_output(wait=True)  # Clear previous plot
            fig = plot_genre_score_distribution(movies_genres, selected_genre)
            fig.show()

    genre_dropdown.observe(update_plot, names='value')

    display(genre_dropdown, output)

    with output:
        fig = plot_genre_score_distribution(movies_genres, genre_dropdown.value)
        fig.show()

# Function to group scores by genres
def group_scores_by_genre(movies_genres):
    groups = [group['Score'].values for _, group in movies_genres.groupby('Movie genres')]
    return groups

# Function to perform the Shapiro-Wilk test for normality
def shapiro_wilk_test(groups):
    print("\nPerforming Shapiro-Wilk test for normality.")
    for i, group in enumerate(groups):
        if len(group) >= 3:  # Shapiro requires at least 3 data points
            stat, p = shapiro(group)
            #print("Performing Shapiro-Wilk test for normality")
            #print(f"Genre {i+1} (size={len(group)}) : W={stat:.3f}, p-value={p:.3f}")
        else:
            print("Not enough data for Shapiro-Wilk test.")

# Function to perform Levene's test for homogeneity of variances
def levenes_test(groups):

    if len(groups) > 1:  # Levene requires at least two groups
        stat, p = levene(*groups)
        print("\nPerforming Levene's test for homogeneity of variances.")
        #print(f"Statistic={stat:.3f}, p-value={p:.3f}")
    else:
        print("\nNot enough groups to perform Levene's test.")

# Function to perform either ANOVA or Kruskal-Wallis test based on assumptions
def perform_statistical_test(groups):
    if all([len(group) >= 3 for group in groups]) and len(groups) > 1:
        normality_passed = all([shapiro(group)[1] > 0.05 for group in groups])
        # Homogeneity of variances check (Levene's test) assumed to be already done
        if normality_passed:
            # Use ANOVA if normality is satisfied
            stat, p = f_oneway(*groups)
            print("\n Normality condition satisfied, performed ANOVA Test.")
            #print(f"F-statistic={stat:.3f}, p-value={p:.3f}")
        else:
            # Use Kruskal-Wallis if assumptions are violated
            stat, p = kruskal(*groups)
            print("\nAssumptions are violated, performed Kruskal-Wallis Test (non-parametric alternative to ANOVA).")
            #print(f"Statistic={stat:.3f}, p-value={p:.3f}")
    else:
        print("\nNot enough valid data or groups to perform statistical tests.")

# Main function that ties everything together
def analyze_movie_genres(movies_genres):
  
    # Step 1: Group scores by genres
    groups = group_scores_by_genre(movies_genres)

    # Step 2: Verify assumptions
    print("\nChecking Assumptions")

    # 2.1 Normality Test
    shapiro_wilk_test(groups)

    # 2.2 Homogeneity of Variances Test (Levene)
    levenes_test(groups)

    # Step 3: Perform statistical test (ANOVA or Kruskal-Wallis)
    perform_statistical_test(groups)

# Function to perform Dunn's test and visualize the results with log-transformed p-values
def dunn_test_and_plot(movies_genres):
    copper_colormap = cm.get_cmap('copper', 256)  # 256 niveaux
    copper_colorscale = [
        [i / 255, f"rgb({int(255 * r)}, {int(255 * g)}, {int(255 * b)})"]
        for i, (r, g, b, _) in enumerate(copper_colormap(np.linspace(0, 1, 256)))
    ]
    # Perform Dunn's test
    dunn_results = sp.posthoc_dunn(movies_genres, val_col='Score', group_col='Movie genres', p_adjust='bonferroni')
    
    # Transform p-values to -log10 scale
    log_dunn_results = -np.log10(dunn_results)  # Use -log10 to make small p-values more prominent

    # Prepare hover text showing genre names and p-values
    hover_text = []
    for i in range(len(dunn_results)):
        row_text = []
        for j in range(len(dunn_results.columns)):
            genre1 = dunn_results.index[i]
            genre2 = dunn_results.columns[j]
            p_value = dunn_results.iloc[i, j]
            row_text.append(f"{genre1} vs {genre2}<br>p-value: {p_value:.4e}")
        hover_text.append(row_text)

    # Create the heatmap
    heatmap = go.Figure(
        data=go.Heatmap(
            z=log_dunn_results.values,
            x=dunn_results.columns,
            y=dunn_results.index,
            text=hover_text,
            hoverinfo="text",
            colorscale=copper_colorscale,  
            zmin=0,                # Min value for color range
            zmax=np.nanmax(log_dunn_results.values),  # Max value for color range
            colorbar=dict(title="-log10(p-value)"),
        )
    )

    # Update layout with title, axes, and size
    heatmap.update_layout(
        title="Dunn's Test Pairwise Comparisons (Log Scale)",
        xaxis=dict(title="Movie Genres", tickangle=45),
        yaxis=dict(title="Movie Genres"),
        height=800, width=800,
    )

    # Show the heatmap
    heatmap.show()

def balance_genres_with_bootstrap(movies_genres, n_samples=500):
    balanced_data = []
    for genre, group in movies_genres.groupby('Movie genres'):
        scores = group['Score'].values
        if len(scores) > n_samples:
            # Bootstrap for large genres
            resampled_scores = np.random.choice(scores, size=n_samples, replace=True)
        else:
            # Keep original data for smaller genres
            resampled_scores = scores
        balanced_data.append(pd.DataFrame({'Movie genres': genre, 'Score': resampled_scores}))
    return pd.concat(balanced_data, ignore_index=True)

def dunn_test_with_balanced_data(movies_genres, n_samples=500):
    balanced_movies_genres = balance_genres_with_bootstrap(movies_genres, n_samples)
    analyze_movie_genres(balanced_movies_genres)
    dunn_test_and_plot(balanced_movies_genres)