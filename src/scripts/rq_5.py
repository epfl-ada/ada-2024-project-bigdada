import pandas as pd
import plotly.express as px

# Function to load and preprocess the dataset
def load_and_preprocess(file_path):
    """
    Load the movie dataset and preprocess by exploding the 'Movie countries' column.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    movies_df = pd.read_csv(file_path, sep='\t')
    movies_df = movies_df.assign(Country=movies_df['Movie countries'].str.split(", ")).explode('Country')
    return movies_df

# Function to create a bar chart for average scores by country
def plot_average_scores_by_country(movies_df, min_films=120):
    """
    Create a bar chart showing average scores by country for countries with a minimum number of films.

    Args:
        movies_df (pd.DataFrame): Preprocessed DataFrame.
        min_films (int): Minimum number of films required for a country to be included.
    """
    # Calculate statistics by country
    country_stats = movies_df.groupby('Movie countries').agg(
        Average_Score=('Score', 'mean'),
        Film_Count=('Score', 'size')
    ).sort_values(by='Average_Score', ascending=False)

    # Filter countries with at least min_films
    country_stats_filtered = country_stats[country_stats['Film_Count'] >= min_films]

    # Create bar plot
    fig = px.bar(
        country_stats_filtered.reset_index(),
        x='Average_Score',
        y='Movie countries',
        color='Average_Score',
        orientation='h',
        title='Average Movie Score by Country',
        labels={'Average_Score': 'Average Score', 'Movie countries': 'Movie countries'},
        height=1000,
        category_orders={"Country": country_stats_filtered.sort_values('Average_Score', ascending=False).index.tolist()}
    )

    # Improve layout
    fig.update_layout(
        yaxis=dict(title='Country', automargin=True),
        xaxis=dict(title='Average Score'),
        coloraxis_colorbar=dict(title='Score'),
        template='plotly_white'
    )

    # Show the plot
    fig.show()

# Function to create a box plot for scores by country
def plot_scores_distribution_by_country(movies_df, min_films=120):
    """
    Create a box plot showing the distribution of scores by country for countries with a minimum number of films.

    Args:
        movies_df (pd.DataFrame): Preprocessed DataFrame.
        min_films (int): Minimum number of films required for a country to be included.
    """
    # Filter countries with at least min_films
    country_counts = movies_df['Movie countries'].value_counts()
    valid_countries = country_counts[country_counts >= min_films].index
    filtered_movies = movies_df[movies_df['Movie countries'].isin(valid_countries)]

    # Create box plot
    fig = px.box(
        filtered_movies,
        x='Movie countries',
        y='Score',
        points='all',
        title='Happy Ending Scores by Country',
        labels={'Score': 'Score', 'Movie countries': 'Movie countries'},
        color='Movie countries',
    )

    # Improve layout
    fig.update_layout(
        xaxis_title='Movie countries',
        yaxis_title='Score',
        xaxis_tickangle=90,
        showlegend=False,
        height=600
    )

    # Show the plot
    fig.show()

