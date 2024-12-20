# This is for research question 4.
# 
# Is there a correlation between a movie’s ending and its success (ratings, box office revenue, etc.)?
# 
# We will explore whether happy or tragic endings have any impact on a movie's popularity or financial performance.
# 

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import plotly.express as px


# path
DATA_FOLDER = '../data/'
MOVIE_DATASET = DATA_FOLDER + 'movies_dataset_final.tsv'


def get_analysis(MOVIE_DATASET):
    # Dataset loading
    movies = pd.read_csv(MOVIE_DATASET, sep='\t')

    # drop movies without vote average or revenue
    movies_filtered = movies.dropna(subset=['vote_average', 'revenue'])

    # drop movies with revenue less than 1000
    movies_filtered = movies_filtered[movies_filtered['revenue'] > 1000]
    return movies_filtered

# plot score vs revenue
def get_score_vs_revenue(movies_filtered, plt=True):
    if plt:
        plt.scatter(movies_filtered['Score'], movies_filtered['revenue'])
        plt.xlabel('Score')
        plt.ylabel('Revenue')
        plt.title('Score vs Revenue')
        plt.show()
    else:
        fig = px.scatter(
            movies_filtered,
            x='Score',
            y='revenue',
            title='Score vs Revenue',
            labels={
                'Score': 'Score',
                'revenue': 'Revenue'
            }
        )
        fig.show()


# plot score vs vote average
def get_score_vs_vote_average(movies_filtered, plt=True):
    if plt:
        plt.scatter(movies_filtered['Score'], movies_filtered['vote_average'])
        plt.xlabel('Score')
        plt.ylabel('Vote Average')
        plt.title('Score vs Vote Average')
        plt.show()
    else:
        fig = px.scatter(
            movies_filtered,
            x='Score',
            y='vote_average',
            title='Score vs Vote Average',
            labels={
                'Score': 'Score',
                'vote_average': 'Vote Average'
            }
        )
        fig.show()



# use t test to test if the average revenue is different for movies with low score and movies with high score
def test_revenue(movies_filtered):
    movies_score_low_revenue = movies_filtered[movies_filtered['Score'] < -0.25]['revenue'].astype(float)
    movies_score_high_revenue = movies_filtered[movies_filtered['Score'] > 0.25]['revenue'].astype(float)

    t_stat, p_value = ttest_ind(movies_score_low_revenue, movies_score_high_revenue)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    if p_value < 0.05:
        print('The difference is statistically significant')
        return True
    else:
        print('The difference is not statistically significant')
        return False
    


# use t test to test if the average vote average is different for movies with low score and movies with high score
def test_vote_average(movies_filtered):
    movies_score_low_vote_average = movies_filtered[movies_filtered['Score'] < -0.25]['vote_average'].astype(float)
    movies_score_high_vote_average = movies_filtered[movies_filtered['Score'] > 0.25]['vote_average'].astype(float)

    t_stat, p_value = ttest_ind(movies_score_low_vote_average, movies_score_high_vote_average)

    print('t-statistic:', t_stat)
    print('p-value:', p_value)

    if p_value < 0.05:
        print('The difference is statistically significant')
        return True
    else:
        print('The difference is not statistically significant')
        return False



