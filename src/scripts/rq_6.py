#!/usr/bin/env python
# coding: utf-8

# # Research question 6
# 
# 

# ### Do budget and production scale affect the type of ending chosen? Exploring whether high-budget films tend to favor certain endings (e.g., happy endings for wider audience appeal) could reveal if financial considerations impact storytelling choices.
# 
# This notebook presents initial observations and is not intended to represent the final conclusions.
# 

# ##### Importations

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from matplotlib import cm
import numpy as np
import plotly.express as px
import ast
from scipy.stats import shapiro, levene, f_oneway, kruskal
import networkx as nx
import scikit_posthocs as sp


copper_colormap = cm.get_cmap('copper', 256)  # 256 niveaux
copper_colorscale = [
    [i / 255, f"rgb({int(255 * r)}, {int(255 * g)}, {int(255 * b)})"]
    for i, (r, g, b, _) in enumerate(copper_colormap(np.linspace(0, 1, 256)))
]


# In[2]:


# path
DATA_FOLDER = 'src/data/'
MOVIE_DATASET = DATA_FOLDER + 'movies_dataset_final.tsv'

# Dataset loading
movies = pd.read_csv(MOVIE_DATASET, sep='\t')


# Remove movies with missing values for budget

# In[3]:


# Count rows where 'budget' is NaN or 0
missing_or_zero_count = movies[(movies['Budget'].isnull()) | (movies['Budget'] == 0)].shape[0]
#print(f"Number of movies with missing or zero budget: {missing_or_zero_count}")

# Calculate the percentage of these rows
percentage_missing_or_zero = (missing_or_zero_count / len(movies)) * 100
#print(f"Percentage of movies with missing or zero budget: {percentage_missing_or_zero:.2f}%")

# Remove rows where 'budget' is NaN or 0
movies = movies[(movies['Budget'].notnull()) & (movies['Budget'] > 0)]

# Verify removal
remaining_rows = len(movies)
#print(f"Number of rows remaining after removal: {remaining_rows}")



# ### Statistics

# In[4]:


# Calculate the correlation between budget and score
correlation = movies['Budget'].corr(movies['Score'])
print(f"Correlation between budget and score: {correlation:.2f}")


# In[ ]:


# Create a scatter plot to visualize the relationship between the budget and the presence of a happy ending
fig = px.scatter(movies,
                 x='Budget',
                 y='Score',
                 title="Budget vs. Happy Ending",
                 labels={'Budget': 'Budget (in millions)', 'Score': 'Happy Ending'},
                 color='Score',  # Color based on Happy_Ending
                 color_discrete_map={True: '#d1b8a1', False: '#f2e4d5'},  # Pastel colors
                 hover_data=['Title', 'Score'])  # Show additional information on hover

# Adjust the size of the graph
fig.update_layout(
    width=800,
    height=600,
    title=dict(font=dict(size=18)),
    title_x=0.5,
)

fig.show()


# 
# Statistical tests are essential for determining whether the observed differences in production scores are significant. Our methodology follows a rigorous three-step process. First, the **Shapiro-Wilk test for normality** ensures that each group’s scores follow a normal distribution, a key assumption for parametric tests like ANOVA. Next, the **Levene’s test for homogeneity of variances** checks whether the variances between groups are equivalent, another prerequisite for ANOVA. If these assumptions are not met, a non-parametric test, such as the **Kruskal-Wallis test**, is employed as a robust alternative. This approach ensures reliable results by adapting to the specific characteristics of the data.
# 

# In[6]:


# Load the dataset
def load_dataset(filepath, sep='\t'):
    try:
        return pd.read_csv(filepath, sep=sep)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# Extract production names from a list
def extract_production_names(production_list):
    if isinstance(production_list, str):
        try:
            production_list = ast.literal_eval(production_list)
        except (ValueError, SyntaxError):
            return []
    if isinstance(production_list, list):
        return [item['name'] for item in production_list if isinstance(item, dict) and 'name' in item]
    return []

# Clean production names (e.g., remove extra spaces and make lowercase)
def clean_production_name(name):
    return name.strip().lower() if isinstance(name, str) else name

# Filter groups with at least three values
def filter_valid_groups(df, column='production_names', min_size=3):
    return df.groupby(column).filter(lambda x: len(x) >= min_size)

# Normality tests (Shapiro-Wilk)
def test_normality(groups):
    print("\nNormality Test (Shapiro-Wilk):")
    normality_results = []
    for i, group in enumerate(groups):
        if len(group) >= 3:
            stat, p = shapiro(group)
            normality_results.append(p > 0.05)
            #print(f"Group {i+1} (size={len(group)}): W={stat:.3f}, p-value={p:.3f}")
        else:
            normality_results.append(False)
            print(f"Group {i+1}: not enough data for Shapiro test.")
    return normality_results

# Variance homogeneity test (Levene)
def test_variance_homogeneity(groups):
    if len(groups) > 1:
        stat, p = levene(*groups)
        print("\nVariance Homogeneity Test (Levene):")
        print(f"Statistic={stat:.3f}, p-value={p:.3f}")
        return p > 0.05
    print("\nNot enough groups to perform Levene's test.")
    return False

# Statistical test (ANOVA or Kruskal-Wallis)
def perform_statistical_test(groups, normality, homogeneity):
    if len(groups) > 1 and all(len(group) >= 3 for group in groups):
        if all(normality) and homogeneity:
            stat, p = f_oneway(*groups)
            print("\nANOVA Test:")
            print(f"F-statistic={stat:.3f}, p-value={p:.3f}")
        else:
            stat, p = kruskal(*groups)
            print("\nKruskal-Wallis Test (non-parametric alternative):")
            print(f"Statistic={stat:.3f}, p-value={p:.14f}")
            if p< 0.05:
                print("\nThere is a significant difference of score between productions")

    else:
        print("\nNot enough data or valid groups to perform statistical tests.")

# Filter productions with a minimum group size
def filter_productions_by_size(valid_groups, min_size=50):
    group_sizes = valid_groups.groupby('production_names').size()
    valid_productions = group_sizes[group_sizes > min_size].index
    return valid_groups[valid_groups['production_names'].isin(valid_productions)]




def plot_significant_comparisons(graph_data):
    
    nude_brown = "#D2B48C"

    # Create a graph from the significant comparisons
    G = nx.Graph()
    for row in graph_data.index:
        for col in graph_data.columns:
            if pd.notna(graph_data.loc[row, col]):  # Only consider significant comparisons
                p_value = graph_data.loc[row, col]
                G.add_edge(row, col, weight=p_value)

    # Define node positions and plot the graph
    pos = nx.spring_layout(G)  # Spring layout for better spacing
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=nude_brown)

    # Draw edges with thickness based on p-value significance
    edges = G.edges(data=True)
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=[(u, v) for u, v, d in edges], 
        width=[2 if d['weight'] < 0.01 else 1 for _, _, d in edges],
        edge_color='gray'
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='black')

    # Add a title
    plt.title("Significant Comparisons Between Productions (Network Graph)", fontsize=16)
    
    # Show the plot (optional)
    plt.show()



# In[7]:


# Loading the dataset
movies = load_dataset(MOVIE_DATASET)
if movies is None:
    raise ValueError("Error while loading the dataset")

# Extracting production names
movies['production_names'] = movies['Production'].apply(extract_production_names)

# Transforming into a long format table
movies_long = movies.explode('production_names')

movies_long['production_names'] = movies_long['production_names'].apply(clean_production_name)

# Filtering valid groups
valid_groups = filter_valid_groups(movies_long, column='production_names')
if valid_groups.empty:
    raise ValueError("No valid groups (size ≥ 3) found for statistical tests.")

# Calculating the average scores per production
production_scores = valid_groups.groupby('production_names')['Score'].mean().reset_index()
production_scores.rename(columns={'Score': 'avg_Score'}, inplace=True)

# Preparing groups for statistical tests
group_prod = valid_groups.groupby('production_names')

# Retrieve the index of Warner Bros productions (in production_scores)
warner_bros_index = group_prod.filter(lambda group: group['production_names'].str.contains("warner bros. cartoons", case=False).any()).index
valid_groups = valid_groups.drop(warner_bros_index)

groups = [group['Score'].values for _, group in group_prod]

# Normality and variance homogeneity tests
normality = test_normality(groups)
homogeneity = test_variance_homogeneity(groups)

# Final statistical test
perform_statistical_test(groups, normality, homogeneity)

# Define a minimum threshold for the number of films per group
min_films_per_group = 50  # For example, at least 50 films per group

# Calculate the size of the groups
group_sizes = valid_groups.groupby('production_names').size()

# Filter groups with at least `min_films_per_group` films
valid_groups = valid_groups[valid_groups['production_names'].isin(group_sizes[group_sizes >= min_films_per_group].index)]

# Create lists for scores and group labels
scores = []
labels = []

# Populate the lists with scores and labels
for production, group in valid_groups.groupby('production_names'):
    scores.extend(group['Score'].values)
    labels.extend([production] * len(group))

# Create a DataFrame with scores and labels
df = pd.DataFrame({
    'score': scores,
    'production_names': labels
})

# Apply Dunn's test with Bonferroni correction for p-values
dunn_results = sp.posthoc_dunn(df, val_col='score', group_col='production_names', p_adjust='bonferroni')

# Filter significant comparisons (p-value < 0.05) and remove NaNs
significant_comparisons = dunn_results[dunn_results < 0.05]

# Remove NaNs from results
significant_comparisons_clean = significant_comparisons.dropna(how='all').dropna(axis=1, how='all')

# Display significant results as "Production 1 vs Production 2: p-value"
print(f"\nSignificant comparisons (p-value < 0.05) for a threshold of {min_films_per_group} movies per production:")


unique_comparisons = set()

# Avoid duplicates (e.g., comparing A with B and B with A)
for row in significant_comparisons_clean.index:
    for col in significant_comparisons_clean.columns:
        if pd.notna(significant_comparisons_clean.loc[row, col]):  # Consider only non-NaN values
            # Create an ordered pair to avoid duplicates
            comparison_pair = tuple(sorted([row, col]))  # Sort names to avoid duplicates like A vs B and B vs A

            # If the pair is not already in the set, add it and print the result
            if comparison_pair not in unique_comparisons:
                unique_comparisons.add(comparison_pair)

                # Calculate the mean scores for each compared production
                mean_row = valid_groups[valid_groups['production_names'] == row]['Score'].mean()
                mean_col = valid_groups[valid_groups['production_names'] == col]['Score'].mean()
                print(f"{row} vs {col}: p-value = {significant_comparisons_clean.loc[row, col]:.5f}, Mean {row} = {mean_row:.2f}, Mean {col} = {mean_col:.2f}")

plot_significant_comparisons(significant_comparisons_clean)


# In[ ]:


def plot_average_scores_plotly(production_scores):
    fig = px.bar(production_scores,
                 x='production_names',
                 y='avg_Score',
                 title='Average Scores by Production',
                 labels={'production_names': 'Production', 'avg_Score': 'Average Score'},
                 color='production_names')
    fig.update_layout(xaxis_title='Production', yaxis_title='Average Score', xaxis_tickangle=90)
    fig.show()

# Filter productions with more than 20 films and calculate the average scores
valid_groups_filtered = filter_productions_by_size(valid_groups)
production_scores_filtered = valid_groups_filtered.groupby('production_names')['Score'].mean().reset_index()
production_scores_filtered.rename(columns={'Score': 'avg_Score'}, inplace=True)

# Visualization
plot_average_scores_plotly(production_scores_filtered)

def plot_boxplot_scores_plotly(valid_groups):
    fig = px.box(valid_groups,
                 x='production_names',
                 y='Score',
                 title='Score Distribution by Production',
                 labels={'production_names': 'Production', 'Score': 'Score'})
    fig.update_layout(xaxis_title='Production', yaxis_title='Score', xaxis_tickangle=90)
    fig.show()

# Visualization
plot_boxplot_scores_plotly(valid_groups_filtered)


# In[9]:


def interactive_bar_plot():

    # Step 1: Create the Dash app
    app = Dash(__name__, title='bar plot')

    # Step 2: Define the app layout
    app.layout = html.Div([
        # Main container
        html.Div([
            # Main graph
            dcc.Graph(
                id='score-visualization',
                config={'displayModeBar': False},  # Disable the toolbar
                style={'flex': '1'}
            ),
            # Dropdown menu positioned at the top right
            html.Div([
                html.Label("Minimum threshold for productions:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='min_films_threshold',
                    options=[
                        {'label': str(value), 'value': value} for value in [1, 5, 10, 30, 50, 100]
                    ],
                    value=5,  # Default value
                    clearable=False,
                    style={'width': '150px',
                           'textAlign': 'center',
                           'paddingRight': '10px'}
                    
                )
            ], style={
                'position': 'absolute',
                'top': '10px',
                'right': '10px',
                'textAlign': 'right'
            }),
        ], style={'position': 'relative'}),

        # Slider at the bottom
        html.Label("Number of films to display:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
        dcc.Slider(
            id='num_films',
            min=1,
            max=10,
            step=1,
            value=5,
            marks={i: str(i) for i in range(1, 11)}
        ),
    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'padding': '20px'})


    # Step 3: Define the callback to update the graph
    @app.callback(
        Output('score-visualization', 'figure'),
        Input('num_films', 'value'),
        Input('min_films_threshold', 'value')
    )
    
    def create_plot(num_films, min_films_threshold):

        
        valid_groups_filtered = filter_productions_by_size(valid_groups, min_films_threshold)
        if valid_groups_filtered.empty:
            return go.Figure().update_layout(title="No data available for these parameters.")
        production_scores_filtered = valid_groups_filtered.groupby('production_names')['Score'].mean().reset_index()
        production_scores_filtered.rename(columns={'Score': 'avg_Score'}, inplace=True)


        # Sort by average score
        production_scores_sorted = production_scores_filtered.sort_values(by='avg_Score', ascending=False)

        # Select the N productions with the highest and lowest scores
        top_n_productions = production_scores_sorted.head(num_films)
        bottom_n_productions = production_scores_sorted.tail(num_films)

        # Combine the two DataFrames
        combined_productions = pd.concat([top_n_productions, bottom_n_productions])

        # Create a bar plot with Plotly
        fig = go.Figure()

        bar_trace= go.Bar(
            x=combined_productions['production_names'],
            y=combined_productions['avg_Score'],
            marker=dict(color=combined_productions['avg_Score'], colorscale=copper_colorscale),
            hoverinfo='x+y',
            name='Productions',
        )

        layout = go.Layout(
            title=f'Top {num_films} and Bottom {num_films} Productions per Mean Score',
            xaxis_title='Name of the Production',
            yaxis_title='Mean Score',
            barmode='overlay',
            template='plotly_white'
        )


        # Combine traces into a figure
        fig = go.Figure(data=[bar_trace], layout=layout)

        # Save the figure as an interactive HTML file
        fig.write_html("graph.html")
        return fig

    # Step 4: Run the app
    app.server.run(port=8005, host='127.0.0.1')


# In[10]:


interactive_bar_plot()


# Step 1: Create the Dash app
app = Dash(__name__, title='bar plot')

# Step 2: Define the app layout
app.layout = html.Div([
    # Main container
    html.Div([
        # Main graph
        dcc.Graph(
            id='score-visualization',
            config={'displayModeBar': False},  # Disable the toolbar
            style={'flex': '1'}
        ),
        # Dropdown menu positioned at the top right
        html.Div([
            html.Label("Minimum threshold for productions:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='min_films_threshold',
                options=[
                    {'label': str(value), 'value': value} for value in [1, 5, 10, 30, 50, 100]
                ],
                value=5,  # Default value
                clearable=False,
                style={'width': '150px',
                        'textAlign': 'center',
                        'paddingRight': '10px'}
                
            )
        ], style={
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'textAlign': 'right'
        }),
    ], style={'position': 'relative'}),

    # Slider at the bottom
    html.Label("Number of films to display:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
    dcc.Slider(
        id='num_films',
        min=1,
        max=10,
        step=1,
        value=5,
        marks={i: str(i) for i in range(1, 11)}
    ),
], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'padding': '20px'})


# Step 3: Define the callback to update the graph
@app.callback(
    Output('score-visualization', 'figure'),
    Input('num_films', 'value'),
    Input('min_films_threshold', 'value')
)

def create_plot(num_films, min_films_threshold):

    
    valid_groups_filtered = filter_productions_by_size(valid_groups, min_films_threshold)
    if valid_groups_filtered.empty:
        return go.Figure().update_layout(title="No data available for these parameters.")
    production_scores_filtered = valid_groups_filtered.groupby('production_names')['Score'].mean().reset_index()
    production_scores_filtered.rename(columns={'Score': 'avg_Score'}, inplace=True)


    # Sort by average score
    production_scores_sorted = production_scores_filtered.sort_values(by='avg_Score', ascending=False)

    # Select the N productions with the highest and lowest scores
    top_n_productions = production_scores_sorted.head(num_films)
    bottom_n_productions = production_scores_sorted.tail(num_films)

    # Combine the two DataFrames
    combined_productions = pd.concat([top_n_productions, bottom_n_productions])

    # Create a bar plot with Plotly
    fig = go.Figure()

    bar_trace= go.Bar(
        x=combined_productions['production_names'],
        y=combined_productions['avg_Score'],
        marker=dict(color=combined_productions['avg_Score'], colorscale=copper_colorscale),
        hoverinfo='x+y',
        name='Productions',
    )

    layout = go.Layout(
        title=f'Top {num_films} and Bottom {num_films} Productions per Mean Score',
        xaxis_title='Name of the Production',
        yaxis_title='Mean Score',
        barmode='overlay',
        template='plotly_white'
    )


    # Combine traces into a figure
    fig = go.Figure(data=[bar_trace], layout=layout)

    # Save the figure as an interactive HTML file
    fig.write_html("graph.html")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)




# In[ ]:




