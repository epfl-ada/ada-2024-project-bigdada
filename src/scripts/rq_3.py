# This is for research question 3.
# 
# What role do key personnel (actors and directors) play in shaping a movie's ending?
# 
# Do certain actors or directors have a preference for particular types of endings, and do their choices influence the overall predictability of a movie’s outcome?

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os
from scipy.stats import ttest_ind


# path
DATA_FOLDER = '../data/'
MOVIE_DATASET = DATA_FOLDER + 'movies_dataset_final.tsv'

def get_analysis(MOVIE_DATASET):
    # Dataset loading
    movies = pd.read_csv(MOVIE_DATASET, sep='\t')

    # drop movies without director, vote average or revenue
    movies_filtered = movies.dropna(subset=['vote_average', 'revenue', 'director'])
    # drop movies with revenue less than 1000
    movies_filtered = movies_filtered[movies_filtered['revenue'] > 1000]

    # count the number of movies per director
    director_counts = movies_filtered['director'].value_counts()
    # filter out directors with more than 4 movies
    directors = director_counts[director_counts > 4].index
    movies_filtered = movies_filtered[movies_filtered['director'].isin(directors)]
    # count the number of movies per director
    director_counts_filtered = movies_filtered['director'].value_counts()

    # calculate the average score per director, and sort the directors by score
    director_avg_score = movies_filtered.groupby('director')['Score'].mean()
    director_avg_score = director_avg_score.sort_values(ascending=False)

    # calculate the score variance per director, and sort the directors by variance
    director_score_variance = movies_filtered.groupby('director')['Score'].std()
    director_score_variance = director_score_variance.sort_values(ascending=False)

    # calculate the average revenue per director, and sort the directors by revenue
    director_avg_revenue = movies_filtered.groupby('director')['revenue'].mean()
    director_avg_revenue = director_avg_revenue.sort_values(ascending=False)

    # calculate the average vote average per director, and sort the directors by vote average
    director_avg_vote_average = movies_filtered.groupby('director')['vote_average'].mean()
    director_avg_vote_average = director_avg_vote_average.sort_values(ascending=False)

    return director_avg_score, director_score_variance, director_avg_revenue, director_avg_vote_average


def get_director_avg_score(director_avg_score, save_path="../../assets/img/rq3/", use_plt=True):
    if use_plt:
        plt.hist(director_avg_score, bins=10, log=True, color='blue', edgecolor='black')
        plt.title('Average ending score per director')
        plt.xlabel('Average ending score')
        plt.ylabel('Number of directors')
        plt.show()
    else:
        fig = px.histogram(
            director_avg_score, 
            nbins=10,
            title='Average ending score per director',
            log_y=True
        )

        fig.update_layout(
            xaxis_title='Average ending score',
            yaxis_title='Number of directors',
            bargap=0.1,
            showlegend=False
        )

        fig.show()
        fig.write_html(os.path.join(save_path, "director_avg_score.html"))

def get_director_score_variance(director_score_variance, save_path="../../assets/img/rq3/", use_plt=True):
    if use_plt:
        plt.hist(director_score_variance, bins=10, log=True, color='green', edgecolor='black')
        plt.title('Ending score variance per director')
        plt.xlabel('Ending score variance')
        plt.ylabel('Number of directors')
        plt.show()
    else:
        fig = px.histogram(
            director_score_variance, 
            nbins=10,
            title='Ending score variance per director',
            log_y=True
        )

        fig.update_layout(
            xaxis_title='Ending score variance',
            yaxis_title='Number of directors',
            bargap=0.1,
            showlegend=False
        )

        fig.show()
        fig.write_html(os.path.join(save_path, "director_score_variance.html"))

def get_director_avg_score_vs_avg_revenue(director_avg_score, director_score_variance, director_avg_revenue, save_path="../../assets/img/rq3/", use_plt=True):
    if use_plt:
        index = director_avg_score.index
        plt.scatter(director_avg_score, director_score_variance[index], c=np.log10(director_avg_revenue), cmap='viridis')
        plt.colorbar(label='Log10(Average Revenue)')
        plt.title('Average ending score vs ending score variance per director')
        plt.xlabel('Average Ending Score')
        plt.ylabel('Ending Score Variance')
        plt.show()
    else:
        fig = px.scatter(
            pd.DataFrame({
                'Average Ending Score': director_avg_score,
                'Ending Score Variance': director_score_variance,
                'Average Revenue': director_avg_revenue
            }),
            x='Average Ending Score',
            y='Ending Score Variance',
            color=np.log10(director_avg_revenue),
            hover_name=director_avg_score.index,
            color_continuous_scale='Viridis',
            title='Average ending score vs ending score variance per director<br>Color is average revenue'
        )

        fig.update_layout(
            xaxis_title='Average Ending Score',
            yaxis_title='Ending Score Variance',
            coloraxis_colorbar=dict(title='Average Revenue', tickvals=[6, 7, 8, 9], ticktext=['1M', '10M', '100M', '1B'])
        )

        fig.show()
        fig.write_html(os.path.join(save_path, "director_avg_score_vs_score_variance.html"))

def get_director_avg_score_vs_avg_vote_average(director_avg_score, director_score_variance, director_avg_vote_average, save_path="../../assets/img/rq3/", use_plt=True):
    if use_plt:
        index = director_avg_score.index
        scatter = plt.scatter(director_avg_score, director_score_variance[index], c=director_avg_vote_average, cmap='viridis')
        plt.colorbar(scatter, label='Average Vote Average')
        plt.title('Average ending score vs ending score variance per director')
        plt.xlabel('Average Ending Score')
        plt.ylabel('Ending Score Variance')
        plt.show()
    else:
        fig = px.scatter(
            pd.DataFrame({
                'Average Ending Score': director_avg_score,
                'Ending Score Variance': director_score_variance,
                'Average Vote Average': director_avg_vote_average
            }),
            x='Average Ending Score',
            y='Ending Score Variance',
            color='Average Vote Average',
            hover_name=director_avg_score.index,
            color_continuous_scale='Viridis',
            title='Average ending score vs ending score variance per director<br>Color is average vote average'
        )

        fig.update_layout(
            xaxis_title='Average Ending Score',
            yaxis_title='Ending Score Variance',
            coloraxis_colorbar=dict(title='Average Vote Average')
        )

        fig.show()
        fig.write_html(os.path.join(save_path, "director_avg_score_vs_score_variance_vote_average.html"))

# use t test to check if the average score is significantly different between directors with high and low revenue
def t_test_avg_score_on_revenue(director_avg_score, director_avg_revenue):
    high_revenue_directors = director_avg_revenue[director_avg_revenue > 0.1e9].index
    low_revenue_directors = director_avg_revenue[director_avg_revenue < 0.1e9].index

    high_revenue_director_avg_score = director_avg_score[high_revenue_directors].dropna()
    low_revenue_director_avg_score = director_avg_score[low_revenue_directors].dropna()

    t_stat, p_value = ttest_ind(high_revenue_director_avg_score, low_revenue_director_avg_score)

    if p_value < 0.05:
        print('The average score is significantly different between directors with high and low revenue')
        return True
    else:
        print('The average score is not significantly different between directors with high and low revenue')
        return False

# use t test to check if the average score is significantly different between directors with high and low vote average
def t_test_avg_score_on_vote_average(director_avg_score, director_avg_vote_average):
    high_vote_average_directors = director_avg_vote_average[director_avg_vote_average > 6.5].index
    low_vote_average_directors = director_avg_vote_average[director_avg_vote_average < 6.5].index

    high_vote_average_director_avg_score = director_avg_score[high_vote_average_directors].dropna()
    low_vote_average_director_avg_score = director_avg_score[low_vote_average_directors].dropna()

    t_stat, p_value = ttest_ind(high_vote_average_director_avg_score, low_vote_average_director_avg_score)

    if p_value < 0.05:
        print('The average score is significantly different between directors with high and low vote average')
        return True
    else:
        print('The average score is not significantly different between directors with high and low vote average')
        return False

# use t test to check if the score variance is significantly different between directors with high and low revenue
def t_test_score_variance_on_revenue(director_score_variance, director_avg_revenue):
    high_revenue_directors = director_avg_revenue[director_avg_revenue > 0.1e9].index
    low_revenue_directors = director_avg_revenue[director_avg_revenue < 0.1e9].index

    high_revenue_director_score_variance = director_score_variance[high_revenue_directors].dropna()
    low_revenue_director_score_variance = director_score_variance[low_revenue_directors].dropna()

    t_stat, p_value = ttest_ind(high_revenue_director_score_variance, low_revenue_director_score_variance)

    if p_value < 0.05:
        print('The score variance is significantly different between directors with high and low revenue')
        return True
    else:
        print('The score variance is not significantly different between directors with high and low revenue')
        return False

# use t test to check if the score variance is significantly different between directors with high and low vote average
def t_test_score_variance_on_vote_average(director_score_variance, director_avg_vote_average):
    high_vote_average_directors = director_avg_vote_average[director_avg_vote_average > 6.5].index
    low_vote_average_directors = director_avg_vote_average[director_avg_vote_average < 6.5].index

    high_vote_average_director_score_variance = director_score_variance[high_vote_average_directors].dropna()
    low_vote_average_director_score_variance = director_score_variance[low_vote_average_directors].dropna()

    t_stat, p_value = ttest_ind(high_vote_average_director_score_variance, low_vote_average_director_score_variance)

    if p_value < 0.05:
        print('The score variance is significantly different between directors with high and low vote average')
        return True
    else:
        print('The score variance is not significantly different between directors with high and low vote average')
        return False