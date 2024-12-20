import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import pycountry_convert as pc

from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, shapiro, ttest_ind 
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


from ipywidgets import interact, Checkbox

from dash import Dash, dcc, html, Input, Output


def bin_years(df, bin_size):
    dataframe = df.copy()
    dataframe['Year bin'] = (dataframe['Movie release date'] // bin_size) * bin_size

    bin_counts = dataframe['Year bin'].value_counts()
    valid_bins = bin_counts[bin_counts >= 100].index 
    dataframe = dataframe[dataframe['Year bin'].isin(valid_bins)]

    return dataframe

def process_movie_countries(df):
    df = df.copy()
    df['Movie countries'] = df['Movie countries'].str.strip('{}')
    df['is_collaborative'] = df['Movie countries'].str.contains(',')

    movies = df[~df['is_collaborative']]

    return movies

def get_binned_counts(df, bins = 3):
    dataframe = df.copy()
    
    dataframe['Year Bin'] = (dataframe['Movie release date'] // bins) * bins
    top_10_idx = dataframe['Movie countries'].value_counts().head(10).index
    movies_10 = dataframe[dataframe['Movie countries'].isin(top_10_idx)]

    binned_counts = movies_10.groupby(['Year Bin', 'Movie countries']).size().unstack(fill_value=0)

    return binned_counts

def get_binned_counts_continent(df, bins = 3):
    dataframe = df.copy()

    dataframe['Year Bin'] = (dataframe['Movie release date'] // bins) * bins
    binned_counts = dataframe.groupby(['Year Bin', 'Continent']).size().unstack(fill_value=0)

    return binned_counts

def compute_average_score(df, bins):
    dataframe = df.copy()
    
    dataframe['Year Bin'] = (dataframe['Movie release date'] // bins) * bins
    top_10_idx = dataframe['Movie countries'].value_counts().head(10).index
    movies_10 = dataframe[dataframe['Movie countries'].isin(top_10_idx)]
    grouped = movies_10.groupby(['Year Bin', 'Movie countries'])['Score']
    
    stats = grouped.agg(average_score=('mean'), std_dev=('std')).reset_index()

    return stats

def compute_all_average_score_per_continent(df, bins):
    dataframe = df.copy()
    
    dataframe['Year Bin'] = (dataframe['Movie release date'] // bins) * bins
    grouped = dataframe.groupby(['Year Bin', 'Continent'])['Score']
    
    stats = grouped.agg(average_score=('mean'), std_dev=('std')).reset_index()

    return stats

def compute_average_score_per_continent(df, bins, min_nmovies):
    dataframe = df.copy()

    dataframe['Year Bin'] = (dataframe['Movie release date'] // bins) * bins
    grouped = dataframe.groupby(['Year Bin', 'Continent'])['Score']

    stats = grouped.agg(average_score=('mean'),
                        std_dev=('std'), 
                        count=('size')).reset_index()
    
    stats['average_score'] = stats.apply(
                                         lambda row: row['average_score'] if row['count'] >= min_nmovies else 0, 
                                         axis = 1)

    return stats

def get_continent(country):
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country, cn_name_format="default")
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = {
            'AF': 'Africa',
            'AS': 'Asia',
            'EU': 'Europe',
            'NA': 'North America',
            'SA': 'South America',
            'OC': 'Oceania',
            'AN': 'Antarctica'
        }
        return continent_name.get(continent_code, 'Unknown')
    except:
        return 'Unknown'

# -------------------------------------------------------

def visualize_scores(df, x):
    grouped = df.groupby('Year bin')['Score']
    means = grouped.mean()
    medians = grouped.median()
    counts = df['Year bin'].value_counts().sort_index()  # Number of movies per bin

    # Step 2: Visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Line plots for mean and median scores
    ax1.plot(means.index, means.values, marker='o', label='Mean Score', color='b')
    #ax1.plot(medians.index, medians.values, marker='s', linestyle='--', label='Median Score', color='g')
    ax1.set_xlabel('Year Bin')
    ax1.set_ylabel('Happy Ending Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Bar plot for movie counts (secondary axis)
    ax2 = ax1.twinx()
    ax2.bar(counts.index, counts.values, width=x * 0.8, alpha=0.3, color='gray', label='Number of Movies')
    ax2.set_ylabel('Number of Movies', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Titles and legends
    plt.title(f'Evolution of Scores Grouped by {x}-Year Bins with Movie Counts')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
 
def plot_movie_countries_over_time(df, bin_size=5):
    """
    Static plot for movie production over time by country with adjustable bin size.

    Parameters:
    - df: DataFrame containing "Year" and country columns with movie counts.
    - bin_size: Integer value representing the size of the bins (in years).
    """
    # Re-bin the data based on the bin size
    df["Year Bin"] = (df["Movie Release date"] // bin_size) * bin_size
    binned_counts = df.groupby("Year Bin").sum()

    # Plot the results
    plt.figure(figsize=(12, 8))

    for country in binned_counts.columns:
        plt.plot(
            binned_counts.index,
            binned_counts[country],
            label=country,
            marker="o",
            linestyle="-"
        )

    # Add labels and grid
    plt.xlabel("Year Bin", fontsize=14)
    plt.ylabel("Number of Movies", fontsize=14)
    plt.title(f"Movie Production Over Time by Country (Bin Size: {bin_size} Years)", fontsize=16)
    plt.xticks(binned_counts.index, rotation=45)
    plt.grid(visible=True, linestyle="--", alpha=0.6)

    # Add legend
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show() 
    
def plot_countries_time_interactive(df):
    # Step 1: Create the Dash app
    app = Dash("countries_time_plot", title="Interactive Movie Production Over Time")

    # Step 2: Define the app layout
    app.layout = html.Div([
        dcc.Graph(id='country-time-visualization'),
        html.Label(
            "Bin Size (Years):",
            style={
                'margin': '10px',
                'textAlign': 'center',
                'color': 'white',
                'fontSize': '16px'
            }
        ),
        dcc.Slider(
            id='bin-slider',
            min=3,
            max=10,
            step=1,
            value=5,  # Default bin size
            marks={i: str(i) for i in range(1, 11)}
        )
    ])

    # Step 3: Define the callback to update the graph
    @app.callback(
        Output('country-time-visualization', 'figure'),
        Input('bin-slider', 'value')
    )
    def update_graph(bin_size):
        # Re-bin the data based on the slider value
        binned_counts = get_binned_counts(df, bin_size)
        
        # Create the figure
        fig = go.Figure()
        for country in binned_counts.columns:
            fig.add_trace(
                go.Scatter(
                    x=binned_counts.index,  # Year bins
                    y=binned_counts[country],  # Movie counts
                    mode='lines',
                    name=country
                )
            )

        fig.update_layout(
            title=f"Movie Production Over Time by Country (Bin Size: {bin_size} Years)",
            xaxis_title="Year Bin",
            xaxis=dict(
                title="Year Bin",
                tickmode="linear",
                dtick=bin_size,
            ),
            yaxis_title="Number of Movies",
            legend_title="Countries",
            template="plotly_white"
        )

        return fig

    # Step 4: Run the app
    app.run_server(debug=True, use_reloader=False, port=8053)

def plot_continent_time_interactive(binned_counts, bins):
    fig = go.Figure()

    for continent in binned_counts.columns:
        fig.add_trace(
            go.Scatter(
                x=binned_counts.index,  # Year bins
                y=binned_counts[continent],  # Movie counts
                mode='lines',
                name=continent,
            )
        )

    fig.update_layout(
        title="Movie Production Over Time by Continent",
        xaxis_title="Year Bin",
        xaxis=dict(
            title="Year Bin",
            tickmode="linear",
            dtick=bins,
        ),
        yaxis_title="Number of Movies",
        legend_title="Continent",
        template="plotly_white",
    )

    return fig

def plot_average_score_time_per_contient_interactive(stats, bins, std = True):
    fig = go.Figure()

    continents = stats['Continent'].unique()
    for continent in continents:
        continent_stats = stats[stats['Continent'] == continent]

        # Add the line for average score
        fig.add_trace(
            go.Scatter(
                x=continent_stats['Year Bin'],
                y=continent_stats['average_score'],
                mode='lines+markers',
                name=f"Average Score ({continent})",
            )
        )

        if std:
            fig.add_trace(
                go.Scatter(
                    x=continent_stats['Year Bin'].tolist() + continent_stats['Year Bin'].tolist()[::-1],
                    y=(continent_stats['average_score'] + continent_stats['std_dev']).tolist() +
                      (continent_stats['average_score'] - continent_stats['std_dev']).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,200,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"Standard Deviation ({continent})",
                    showlegend=False  # Hide additional legend entries for std deviation
                )
            )

    fig.update_layout(
        title="Average Score Over Time by Country",
        xaxis_title="Year Bin",
        xaxis=dict(
            title="Year Bin",
            tickmode="linear",
            dtick=bins,
        ),
        yaxis_title="Average Score",
        legend_title="Continent",
        template="plotly_white",
    )

    return fig

def interactive_average_score_plot(df):
    # Step 1: Create the Dash app
    app = Dash("average_score_time_plot", title="Interactive Average Score Over Time")

    # Step 2: Define the app layout
    app.layout = html.Div([
        dcc.Graph(id='average-score-visualization'),
        html.Label(
            "Bin Size (Years):",
            style={
                'margin': '10px',
                'textAlign': 'center',
                'color': 'black',
                'fontSize': '16px'
            }
        ),
        dcc.Slider(
            id='bin-slider',
            min=3,
            max=10,
            step=1,
            value=5,  # Default bin size
            marks={i: str(i) for i in range(1, 11)}
        )
    ])

    # Step 3: Define the callback to update the graph
    @app.callback(
        Output('average-score-visualization', 'figure'),
        Input('bin-slider', 'value')
    )
    def update_graph(bin_size):
        # Compute stats based on the current bin size
        stats = compute_average_score(df, bin_size)

        # Initialize figure
        fig = go.Figure()

        countries = stats['Movie countries'].unique()
        for country in countries:
            country_stats = stats[stats['Movie countries'] == country]

            # Add line for average score
            fig.add_trace(
                go.Scatter(
                    x=country_stats['Year Bin'],
                    y=country_stats['average_score'],
                    mode='lines+markers',
                    name=f"Average Score ({country})",
                )
            )

        fig.update_layout(
            title=f"Average Score Over Time by Country (Bin Size: {bin_size} Years)",
            xaxis_title="Year Bin",
            xaxis=dict(
                title="Year Bin",
                tickmode="linear",
                dtick=bin_size,
            ),
            yaxis_title="Average Score",
            legend_title="Countries",
            template="plotly_white"
        )

        return fig

    # Step 4: Run the app
    app.run_server(debug=True, use_reloader=False, port=8054)

def interactive_bin_plot(df):
    # Step 1: Create the Dash app
    app = Dash("bin_plot_app", title='Bin Plot')

    # Step 2: Define the app layout
    app.layout = html.Div([
        dcc.Graph(id='score-visualization'),
         html.Label(
                "Bin Size (Years):",
                style={
                    'margin': '10px',
                    'textAlign': 'center',  # Center-align the text
                    'color': 'white',       # Make the text white
                    'fontSize': '16px'      # Optional: Adjust font size
                }
         ),
        dcc.Slider(
            id='bin-slider',
            min=1,
            max=10,
            step=1,
            value=5,  # Initial bin size
            marks={i: str(i) for i in range(1, 10)}
        )
    ])

    # Step 3: Define the callback to update the graph
    @app.callback(
        Output('score-visualization', 'figure'),
        Input('bin-slider', 'value')
    )
    def update_graph(bin_size):
        # Bin the years using the current bin size
        binned_df = bin_years(df.copy(), bin_size)
        grouped = binned_df.groupby('Year bin')['Score']
        means = grouped.mean()
        medians = grouped.median()
        counts = binned_df['Year bin'].value_counts().sort_index()

        # Create scatter traces for mean and median scores
        mean_trace = go.Scatter(
            x=means.index,
            y=means.values,
            mode='lines+markers',
            name='Mean Score',
            line=dict(color='blue', width=2),
            marker=dict(size=8, symbol='circle')
        )

        median_trace = go.Scatter(
            x=medians.index,
            y=medians.values,
            mode='lines+markers',
            name='Median Score',
            line=dict(dash='dash', color='green', width=2),
            marker=dict(size=8, symbol='square')
        )

        # Create a bar trace for movie counts
        bar_trace = go.Bar(
            x=counts.index,
            y=counts.values,
            name='Number of Movies',
            marker=dict(color='rgba(128, 128, 128, 0.5)', line=dict(color='gray', width=1)),
            yaxis='y2'
        )

        # Layout definition
        layout = go.Layout(
            title=f'Evolution of Scores Grouped by {bin_size}-Year Bins with Movie Counts',
            xaxis=dict(title='Year Bin', tickangle=45),
            yaxis=dict(title='Happy Ending Score', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
            yaxis2=dict(
                title='Number of Movies',
                titlefont=dict(color='gray'),
                tickfont=dict(color='gray'),
                overlaying='y',
                side='right'
            ),
            legend=dict(
                x=0.5,
                y=1.1,
                xanchor='center',
                orientation='h'
            ),
            barmode='overlay',
            template='plotly_white'
        )

        # Combine traces into a figure
        fig = go.Figure(data=[mean_trace, median_trace, bar_trace], layout=layout)
        return fig

    # Step 4: Run the app
    app.run_server(debug=True, use_reloader=False, port=8050)
    
def interactive_bin_plot_html(df, output_file="interactive_bin_plot.html"):
    # Precompute data for all bin sizes
    bin_sizes = list(range(1, 11))  
    precomputed_data = {}
    
    for bin_size in bin_sizes:
        binned_df = bin_years(df.copy(), bin_size)
        grouped = binned_df.groupby('Year bin')['Score']
        means = grouped.mean()
        medians = grouped.median()
        counts = binned_df['Year bin'].value_counts().sort_index()
        precomputed_data[bin_size] = (means, medians, counts)
    
    # Create traces for each bin size
    all_traces = []
    steps = []

    for i, bin_size in enumerate(bin_sizes):
        means, medians, counts = precomputed_data[bin_size]
        
        # Create scatter traces for mean and median scores
        mean_trace = go.Scatter(
            x=means.index,
            y=means.values,
            mode='lines+markers',
            name=f'Mean Score (Bin={bin_size})',
            line=dict(dash='dash', color='rgb(102, 51, 0)', width=2),  # Darker copper tone
            marker=dict(size=8, symbol='square', color='rgb(153, 76, 0)'),
            visible=(i == 0)
        )

        median_trace = go.Scatter(
            x=medians.index,
            y=medians.values,
            mode='lines+markers',
            name=f'Median Score (Bin={bin_size})',
            line=dict(color='rgb(153, 102, 51)', width=2),  # Copper tone
            marker=dict(size=8, symbol='circle', color='rgb(204, 153, 102)'),
            visible=(i == 0)
        )

        # Create a bar trace for movie counts
        bar_trace = go.Bar(
            x=counts.index,
            y=counts.values,
            name=f'Number of Movies (Bin={bin_size})',
            marker=dict(color='rgba(128, 128, 128, 0.5)', line=dict(color='gray', width=1)),
            yaxis='y2',
            visible=(i == 0)
        )

        # Add traces to the list
        all_traces.extend([mean_trace, median_trace, bar_trace])

        # Add a step for the slider
        visibility = [False] * len(bin_sizes) * 3  # Total traces = 3 * len(bin_sizes)
        visibility[i * 3:(i + 1) * 3] = [True, True, True]  # Make only current bin size visible
        steps.append({
            "method": "update",
            "args": [{"visible": visibility},
                     {"title": ''}],
            "label": f"{bin_size}"
        })

    # Define layout with sliders
    layout = go.Layout(
    xaxis=dict(title='Year Bin', tickangle=45),
    yaxis=dict(title='Happy Ending Score', titlefont=dict(color='maroon'), tickfont=dict(color='maroon')),
    yaxis2=dict(
        title='Number of Movies',
        titlefont=dict(color='gray'),
        tickfont=dict(color='gray'),
        overlaying='y',
        side='right'
    ),
    legend=dict(
        x=0.5,
        y=1.1,
        xanchor='center',
        orientation='h'
    ),
    barmode='overlay',
    template='plotly_white',
    sliders=[{
        "active": 0,
        "currentvalue": {"prefix": "Bin Size: "},
        "pad": {"t": 50},
        "steps": steps
    }],
    width=800,  # Adjust the width of the graph
    height=600   # Optional: Adjust the height of the graph
    )

    # Combine traces and layout into a figure
    fig = go.Figure(data=all_traces, layout=layout)
    fig.show()
    
    # Save to HTML
    pio.write_html(fig, file=output_file, full_html=True, include_plotlyjs="cdn")
    print(f"Interactive HTML saved to: {output_file}")

def plot_pvalues_mannwhitneyu(df):
    app = Dash(__name__, title="P-Value and Sample Sizes")

    # Define app layout
    app.layout = html.Div([
        dcc.Graph(id="pvalue-plot"),
        html.Label("Adjust Number of Dates:", style={
            'margin': '10px',
            'textAlign': 'center',
            'color': 'white',
            'fontSize': '16px'
        }),
        dcc.Slider(
            id="threshold-step-slider",
            min=1,
            max=10,
            step=1,
            value=2,  # Default
            marks=None
        ),
        html.Label("Show Sample Sizes?", style={
            'margin': '10px',
            'textAlign': 'center',
            'color': 'white',
            'fontSize': '16px'
        }),
        dcc.Checklist(
            id="toggle-sample-sizes",
            options=[{"label": "", "value": "show"}],
            value=[]  # Default: sample sizes not shown
        ),
    ])

    # Callback to update graph based on slider value and checkbox
    @app.callback(
        Output("pvalue-plot", "figure"),
        [Input("threshold-step-slider", "value"),
         Input("toggle-sample-sizes", "value")]
    )
    def update_plot(granularity, toggle_sample_sizes):
        step = 11 - granularity
        thresholds = list(range(1930, 2014, step))
        p_values = []
        pre_sizes = []
        post_sizes = []

        # Compute p-values and sample sizes for each threshold
        for threshold in thresholds:
            pre_group = df[df["Movie release date"] < threshold]["Score"]
            post_group = df[df["Movie release date"] >= threshold]["Score"]

            # Store sample sizes
            pre_sizes.append(len(pre_group))
            post_sizes.append(len(post_group))

            # Perform Mann-Whitney U test
            stat, p_value = mannwhitneyu(pre_group, post_group, alternative="two-sided")
            p_values.append(p_value)

        # Prepare data for the plot
        thresholds = np.array(thresholds)
        p_values = np.array(p_values)

        # Highlight significant thresholds
        significant_mask = p_values < 0.05

        # Create figure
        fig = go.Figure()

        # Add p-values as a line
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=p_values,
            mode="lines+markers",
            name="P-Value",
            line=dict(shape="spline", color="maroon", width=2),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
        ))

        # Highlight significant points
        if significant_mask.any():
            fig.add_trace(go.Scatter(
                x=thresholds[significant_mask],
                y=p_values[significant_mask],
                mode="markers",
                name="Significant (p < 0.05)",
                marker=dict(size=10, color="firebrick", symbol="diamond"),
                hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
            ))

        # Add a horizontal line at p = 0.05
        fig.add_hline(
            y=0.05,
            line_dash="dot",
            line_color="gray",
            annotation_text="p = 0.05",
            annotation_position="bottom left"
        )

        # Add stacked bar plots for sample sizes if checkbox is checked
        if "show" in toggle_sample_sizes:
            fig.add_trace(go.Bar(
                x=thresholds,
                y=pre_sizes,
                name="Pre-Threshold Group Size",
                marker_color="#D2B48C",  # Light Brown
                opacity=0.4,  # Semi-transparent
                hovertemplate="<b>Threshold:</b> %{x}<br><b>Pre-Group Size:</b> %{y}<extra></extra>",
                yaxis="y2"
            ))
            fig.add_trace(go.Bar(
                x=thresholds,
                y=post_sizes,
                name="Post-Threshold Group Size",
                marker_color="#F4A460",  # Sandy Brown
                opacity=0.4,  # Semi-transparent
                hovertemplate="<b>Threshold:</b> %{x}<br><b>Post-Group Size:</b> %{y}<extra></extra>",
                yaxis="y2"
            ))

        # Update layout
        fig.update_layout(
            title={
                "text": "P-Value and Sample Sizes Analysis",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis=dict(
                title="Threshold Year",
                tickangle=45,
                showgrid=True
            ),
            yaxis=dict(
                title="P-Value",
                showgrid=True,
                tickformat=".2f"
            ),
            yaxis2=dict(
                title="Sample Sizes",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            template="plotly_white",
            height=600,
            width=900,
            barmode="stack",
            legend=dict(
                x=0.5,
                y=-0.2,
                xanchor="center",
                orientation="h"
            )
        )
        
        pio.write_html(fig, file="pvalues_mannwhitneyu.html", auto_open=False)

        return fig

    # Run the app
    app.run_server(debug=True, use_reloader=False, port=8051)
    
def plot_pvalues_without_samples(df, output_file="pvalues_without_samples.html"):
    granularities = list(range(1, 11))  # Granularity from 1 to 10
    precomputed_data = {}
    
    for granularity in granularities:
        step = 11 - granularity
        thresholds = list(range(1930, 2014, step))
        p_values = []

        for threshold in thresholds:
            pre_group = df[df["Movie release date"] < threshold]["Score"]
            post_group = df[df["Movie release date"] >= threshold]["Score"]

            # Perform Mann-Whitney U test
            stat, p_value = mannwhitneyu(pre_group, post_group, alternative="two-sided")
            p_values.append(p_value)

        precomputed_data[granularity] = {
            "thresholds": np.array(thresholds),
            "p_values": np.array(p_values),
        }

    all_traces = []
    steps = []

    for i, granularity in enumerate(granularities):
        data = precomputed_data[granularity]
        thresholds = data["thresholds"]
        p_values = data["p_values"]

        significant_mask = p_values < 0.05

        # Add p-value trace
        all_traces.append(go.Scatter(
            x=thresholds,
            y=p_values,
            mode="lines+markers",
            name=f"P-Value",
            line=dict(shape="spline", color="maroon", width=2),
            marker=dict(size=8, symbol="circle"),
            visible=(i == 0),
            hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
        ))

        # Add significant points trace
        all_traces.append(go.Scatter(
            x=thresholds[significant_mask],
            y=p_values[significant_mask],
            mode="markers",
            name=f"Significant (p < 0.05)",
            marker=dict(size=10, color="firebrick", symbol="diamond"),
            visible=(i == 0),
            hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
        ))

        # Add step for slider
        visibility = [False] * len(granularities) * 2  # 2 traces per granularity
        visibility[i * 2:(i + 1) * 2] = [True, True]  # Show only current traces
        steps.append({
            "method": "update",
            "args": [{"visible": visibility},
                     {"title": f"P-Value Analysis (Granularity={granularity})"}],
            "label": f"{granularity}"
        })

    # Define layout with sliders
    layout = go.Layout(
        xaxis=dict(title="Threshold Year", tickangle=45, showgrid=True),
        yaxis=dict(title="P-Value", showgrid=True, tickformat=".2f"),
        template="plotly_white",
        height=600,
        width=900,
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Granularity: "},
            "pad": {"t": 50},
            "steps": steps
        }],
        legend=dict(
        x=0.5,          # Center the legend horizontally
        y=1.15,         # Place the legend above the plot
        xanchor="center",  # Anchor the horizontal position
        yanchor="top",     # Anchor the vertical position
        orientation="h"    # Horizontal orientation
        ),
    )

    # Combine traces and layout into a figure
    fig = go.Figure(data=all_traces, layout=layout)
    fig.show()
    # Save to HTML
    pio.write_html(fig, file=output_file, full_html=True, include_plotlyjs="cdn")
    print(f"Interactive HTML without sample sizes saved to: {output_file}")

def plot_pvalues_without_samples_matplotlib(df, threshold_granularity, output_file="pvalues_without_samples.png"):
    """
    Create a static plot of p-values using Matplotlib.

    Parameters:
    - df: DataFrame containing "Movie release date" and "Score" columns.
    - threshold_granularity: Integer value to define the granularity of thresholds (1-10).
    - output_file: File path to save the static plot as a PNG file.
    """
    if not (1 <= threshold_granularity <= 10):
        raise ValueError("Threshold granularity must be between 1 and 10.")

    step = 11 - threshold_granularity
    thresholds = list(range(1930, 2014, step))
    p_values = []

    for threshold in thresholds:
        pre_group = df[df["Movie release date"] < threshold]["Score"]
        post_group = df[df["Movie release date"] >= threshold]["Score"]

        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(pre_group, post_group, alternative="two-sided")
        p_values.append(p_value)

    thresholds = np.array(thresholds)
    p_values = np.array(p_values)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot p-values
    plt.plot(thresholds, p_values, marker="o", linestyle="-", color="maroon", label="P-Value")

    # Highlight significant points
    significant_mask = p_values < 0.05
    plt.scatter(
        thresholds[significant_mask],
        p_values[significant_mask],
        color="firebrick",
        label="Significant (p < 0.05)",
        s=80,
        marker="D",
    )

    # Add labels and grid
    plt.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="p = 0.05")
    plt.xlabel("Threshold Year", fontsize=12)
    plt.ylabel("P-Value", fontsize=12)
    plt.title(f"P-Value Analysis (Granularity={threshold_granularity})", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(visible=True, linestyle="--", alpha=0.6)

    # Add legend
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
    plt.show()
   
def plot_pvalues_with_samples(df, output_file="pvalues_with_samples.html"):
    granularities = list(range(1, 11))  # Granularity from 1 to 10
    precomputed_data = {}
    
    for granularity in granularities:
        step = 11 - granularity
        thresholds = list(range(1930, 2014, step))
        p_values = []
        pre_sizes = []
        post_sizes = []

        for threshold in thresholds:
            pre_group = df[df["Movie release date"] < threshold]["Score"]
            post_group = df[df["Movie release date"] >= threshold]["Score"]

            pre_sizes.append(len(pre_group))
            post_sizes.append(len(post_group))

            # Perform Mann-Whitney U test
            stat, p_value = mannwhitneyu(pre_group, post_group, alternative="two-sided")
            p_values.append(p_value)

        precomputed_data[granularity] = {
            "thresholds": np.array(thresholds),
            "p_values": np.array(p_values),
            "pre_sizes": np.array(pre_sizes),
            "post_sizes": np.array(post_sizes),
        }

    all_traces = []
    steps = []

    for i, granularity in enumerate(granularities):
        data = precomputed_data[granularity]
        thresholds = data["thresholds"]
        p_values = data["p_values"]
        pre_sizes = data["pre_sizes"]
        post_sizes = data["post_sizes"]

        significant_mask = p_values < 0.05

        # Add p-value trace
        all_traces.append(go.Scatter(
            x=thresholds,
            y=p_values,
            mode="lines+markers",
            name=f"P-Value",
            line=dict(shape="spline", color="maroon", width=2),
            marker=dict(size=8, symbol="circle"),
            visible=(i == 0),
            hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
        ))

        # Add significant points trace
        all_traces.append(go.Scatter(
            x=thresholds[significant_mask],
            y=p_values[significant_mask],
            mode="markers",
            name=f"Significant (p < 0.05)",
            marker=dict(size=10, color="firebrick", symbol="diamond"),
            visible=(i == 0),
            hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
        ))

        # Add bar traces for sample sizes
        all_traces.append(go.Bar(
            x=thresholds,
            y=pre_sizes,
            name=f"Pre-Threshold Group Size ",
            marker_color="#D2B48C",
            opacity=0.4,
            visible=(i == 0),
            hovertemplate="<b>Threshold:</b> %{x}<br><b>Pre-Group Size:</b> %{y}<extra></extra>",
            yaxis="y2"
        ))
        all_traces.append(go.Bar(
            x=thresholds,
            y=post_sizes,
            name=f"Post-Threshold Group Size",
            marker_color="#F4A460",
            opacity=0.4,
            visible=(i == 0),
            hovertemplate="<b>Threshold:</b> %{x}<br><b>Post-Group Size:</b> %{y}<extra></extra>",
            yaxis="y2"
        ))

        # Add step for slider
        visibility = [False] * len(granularities) * 4  # 4 traces per granularity
        visibility[i * 4:(i + 1) * 4] = [True, True, True, True]
        steps.append({
            "method": "update",
            "args": [{"visible": visibility},
                     {"title": ''}],
            "label": f"{granularity}"
        })

    # Define layout with sliders
    layout = go.Layout(
        xaxis=dict(title="Threshold Year", tickangle=45, showgrid=True),
        yaxis=dict(title="P-Value", showgrid=True, tickformat=".2f"),
        yaxis2=dict(
            title="Sample Sizes",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        height=600,
        width=900,
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Granularity: "},
            "pad": {"t": 50},
            "steps": steps
        }],
        legend=dict(
        x=0.5,          # Center the legend horizontally
        y=1.15,         # Place the legend above the plot
        xanchor="center",  # Anchor the horizontal position
        yanchor="top",     # Anchor the vertical position
        orientation="h"    # Horizontal orientation
        )
    )

    # Combine traces and layout into a figure
    fig = go.Figure(data=all_traces, layout=layout)
    fig.show()
    # Save to HTML
    pio.write_html(fig, file=output_file, full_html=True, include_plotlyjs="cdn")
    print(f"Interactive HTML with sample sizes saved to: {output_file}")

def plot_pvalues_mannwhitneyu_bootstrap(df, thresholds, num_samples=300, n_bootstraps=100):
    p_values = []
    cohen_ds = []
    cohen_ds_ci = []

    for threshold in thresholds:
        pre_group = df[df["Movie release date"] < threshold]["Score"].dropna()
        post_group = df[df["Movie release date"] >= threshold]["Score"].dropna()

        # Adjust sampling to use all available data if groups are smaller than num_samples
        if len(pre_group) < num_samples or len(post_group) < num_samples:
            available_samples = min(len(pre_group), len(post_group))
            pre_sample = np.random.choice(pre_group, available_samples, replace=False)
            post_sample = np.random.choice(post_group, available_samples, replace=False)
        else:
            pre_sample = np.random.choice(pre_group, num_samples, replace=False)
            post_sample = np.random.choice(post_group, num_samples, replace=False)

        # Bootstrap resampling
        bootstrap_pvals, bootstrap_ds = [], []
        for _ in range(n_bootstraps):
            resampled_pre = np.random.choice(pre_sample, len(pre_sample), replace=True)
            resampled_post = np.random.choice(post_sample, len(post_sample), replace=True)
            _, pval = mannwhitneyu(resampled_pre, resampled_post, alternative="two-sided")
            bootstrap_pvals.append(pval)
            d = cohens_d(resampled_pre, resampled_post)
            bootstrap_ds.append(d)

        # Store results
        p_values.append(np.mean(bootstrap_pvals))
        cohen_ds.append(np.mean(bootstrap_ds))
        cohen_ds_ci.append((np.percentile(bootstrap_ds, 2.5), np.percentile(bootstrap_ds, 97.5)))

    thresholds_arr = np.array(thresholds)
    p_values = np.array(p_values)
    cohen_ds = np.array(cohen_ds)

    fig = go.Figure()

    # Plot p-values
    fig.add_trace(go.Scatter(
        x=thresholds_arr,
        y=p_values,
        mode="lines+markers",
        name="P-Value",
        line=dict(shape="spline", color="steelblue", width=2),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
    ))

    # Plot Cohen's d with CI as shaded areas
    for i, (d, ci) in enumerate(zip(cohen_ds, cohen_ds_ci)):
        fig.add_trace(go.Scatter(
            x=[thresholds_arr[i], thresholds_arr[i]],
            y=[ci[0], ci[1]],
            mode="lines",
            line=dict(color="lightcoral", width=2),
            showlegend=False,
            yaxis="y2"
        ))

    # Plot Cohen's d
    fig.add_trace(go.Scatter(
        x=thresholds_arr,
        y=cohen_ds,
        mode="lines+markers",
        name="Cohen's d",
        line=dict(shape="spline", color="firebrick", width=2),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>Threshold:</b> %{x}<br><b>Cohen's d:</b> %{y:.2f}<extra></extra>",
        yaxis="y2"
    ))

    # Layout updates
    fig.update_layout(
        title="P-Value and Effect Size Plot with Confidence Intervals",
        xaxis=dict(title="Threshold Year", tickangle=45, showgrid=True),
        yaxis=dict(title="P-Value", showgrid=True),
        yaxis2=dict(title="Cohen's d", overlaying="y", side="right"),
        template="plotly_white",
        height=600,
        width=900,
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")
    )

    # Show the plot
    fig.show()

def rank_biserial_correlation(pre_group, post_group):
    n1 = len(pre_group)
    n2 = len(post_group)
    u_stat, _ = mannwhitneyu(pre_group, post_group, alternative="two-sided")
    return 1 - (2 * u_stat) / (n1 * n2)

def plot_pvalues_mannwhitneyu_bootstrap_rbc(df, thresholds, num_samples=300, n_bootstraps=100):
    p_values = []
    rbcs = []
    rbcs_ci = []

    for threshold in thresholds:
        pre_group = df[df["Movie release date"] < threshold]["Score"].dropna()
        post_group = df[df["Movie release date"] >= threshold]["Score"].dropna()

        # Adjust sampling to use all available data if groups are smaller than num_samples
        if len(pre_group) < num_samples or len(post_group) < num_samples:
            available_samples = min(len(pre_group), len(post_group))
            pre_sample = np.random.choice(pre_group, available_samples, replace=False)
            post_sample = np.random.choice(post_group, available_samples, replace=False)
        else:
            pre_sample = np.random.choice(pre_group, num_samples, replace=False)
            post_sample = np.random.choice(post_group, num_samples, replace=False)

        # Bootstrap resampling
        bootstrap_pvals, bootstrap_rbcs = [], []
        for _ in range(n_bootstraps):
            resampled_pre = np.random.choice(pre_sample, len(pre_sample), replace=True)
            resampled_post = np.random.choice(post_sample, len(post_sample), replace=True)
            u_stat, pval = mannwhitneyu(resampled_pre, resampled_post, alternative="two-sided")
            bootstrap_pvals.append(pval)
            
            # Compute RBC directly from U statistic
            n1 = len(resampled_pre)
            n2 = len(resampled_post)
            rbc = 1 - (2 * u_stat) / (n1 * n2)
            bootstrap_rbcs.append(rbc)

        # Store results
        p_values.append(np.mean(bootstrap_pvals))
        rbcs.append(np.mean(bootstrap_rbcs))
        rbcs_ci.append((np.percentile(bootstrap_rbcs, 2.5), np.percentile(bootstrap_rbcs, 97.5)))

    thresholds_arr = np.array(thresholds)
    p_values = np.array(p_values)
    rbcs = np.array(rbcs)

    fig = go.Figure()

    # Plot p-values
    fig.add_trace(go.Scatter(
        x=thresholds_arr,
        y=p_values,
        mode="lines+markers",
        name="P-Value",
        line=dict(shape="spline", color="chocolate", width=2),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
    ))

    # Plot Rank-Biserial Correlation with CI as shaded areas
    for i, (rbc, ci) in enumerate(zip(rbcs, rbcs_ci)):
        fig.add_trace(go.Scatter(
            x=[thresholds_arr[i], thresholds_arr[i]],
            y=[ci[0], ci[1]],
            mode="lines",
            line=dict(color="lightcoral", width=2),
            showlegend=False,
            yaxis="y2"
        ))
    fig.add_trace(go.Scatter(
    x=thresholds_arr,
    y=[0.05] * len(thresholds_arr),  # Horizontal line at p = 0.05
    mode="lines",
    line=dict(color="brown", dash="dash"),
    name="p = 0.05",
    hoverinfo="skip"
    ))

    # Plot Rank-Biserial Correlation
    fig.add_trace(go.Scatter(
        x=thresholds_arr,
        y=rbcs,
        mode="lines+markers",
        name="Rank-Biserial Correlation",
        line=dict(shape="spline", color="firebrick", width=2),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>Threshold:</b> %{x}<br><b>RBC:</b> %{y:.2f}<extra></extra>",
        yaxis="y2"
    ))

    # Layout updates
    fig.update_layout(
        title="P-Value and Rank-Biserial Correlation Plot with Confidence Intervals",
        xaxis=dict(title="Threshold Year", tickangle=45, showgrid=True),
        yaxis=dict(title="P-Value", showgrid=True),
        yaxis2=dict(title="Rank-Biserial Correlation", overlaying="y", side="right"),
        template="plotly_white",
        height=600,
        width=900,
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")
    )

    # Show the plot
    fig.show()
    pio.write_html(fig, file="bootstrap_rbc.html", full_html=True, include_plotlyjs="cdn")
    print("Interactive HTML with sample sizes saved to: bootstrap_rbc")

def plot_distributions_by_thresholds(df, thresholds, num_samples=None):
    # Define a copper colormap
    cmap = cm.get_cmap('copper')
    norm = mcolors.Normalize(vmin=min(thresholds), vmax=max(thresholds))

    # Create subplots
    num_rows = len(thresholds)
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, subplot_titles=[f"Threshold: {t}" for t in thresholds])

    for i, threshold in enumerate(thresholds):
        pre_group = df[df["Movie release date"] < threshold]["Score"].dropna()
        post_group = df[df["Movie release date"] >= threshold]["Score"].dropna()

        # Balance groups if num_samples is specified
        if num_samples is not None:
            available_samples = min(len(pre_group), len(post_group), num_samples)
            pre_group = np.random.choice(pre_group, available_samples, replace=False)
            post_group = np.random.choice(post_group, available_samples, replace=False)

        # Create histograms for pre and post groups
        pre_hist = np.histogram(pre_group, bins=20, range=(df["Score"].min(), df["Score"].max()))
        post_hist = np.histogram(post_group, bins=20, range=(df["Score"].min(), df["Score"].max()))

        # Normalize histograms
        pre_hist_vals = pre_hist[0] / len(pre_group)
        post_hist_vals = post_hist[0] / len(post_group)
        bin_edges = pre_hist[1]

        # Define colors from colormap
        color_pre = mcolors.to_hex(cmap(norm(threshold - 5)))
        color_post = mcolors.to_hex(cmap(norm(threshold + 5)))

        # Add traces for pre and post groups
        fig.add_trace(
            go.Bar(
                x=bin_edges[:-1],
                y=pre_hist_vals,
                name=f"Pre {threshold}",
                marker_color=color_pre,
                opacity=0.7,
                hovertemplate="<b>Score:</b> %{x}<br><b>Density:</b> %{y:.3f}<extra></extra>",
            ),
            row=i + 1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=bin_edges[:-1],
                y=post_hist_vals,
                name=f"Post {threshold}",
                marker_color=color_post,
                opacity=0.7,
                hovertemplate="<b>Score:</b> %{x}<br><b>Density:</b> %{y:.3f}<extra></extra>",
            ),
            row=i + 1, col=1
        )

    # Update layout
    fig.update_layout(
        title="Distributions of Scores Before and After Thresholds",
        height=300 * len(thresholds),  # Dynamic height based on number of thresholds
        width=900,
        template="plotly_white",
        showlegend=False
    )

    # Show the plot
    fig.show()
    
def plot_pvalues_welch_bootstrap_cohensd(df, thresholds, num_samples=300, n_bootstraps=100):
    p_values = []
    cohens_d = []
    cohens_d_ci = []

    for threshold in thresholds:
        pre_group = df[df["Movie release date"] < threshold]["Score"].dropna()
        post_group = df[df["Movie release date"] >= threshold]["Score"].dropna()

        # Adjust sampling to use all available data if groups are smaller than num_samples
        if len(pre_group) < num_samples or len(post_group) < num_samples:
            available_samples = min(len(pre_group), len(post_group))
            pre_sample = np.random.choice(pre_group, available_samples, replace=False)
            post_sample = np.random.choice(post_group, available_samples, replace=False)
        else:
            pre_sample = np.random.choice(pre_group, num_samples, replace=False)
            post_sample = np.random.choice(post_group, num_samples, replace=False)

        # Bootstrap resampling
        bootstrap_pvals, bootstrap_cohensd = [], []
        for _ in range(n_bootstraps):
            resampled_pre = np.random.choice(pre_sample, len(pre_sample), replace=True)
            resampled_post = np.random.choice(post_sample, len(post_sample), replace=True)
            t_stat, pval = ttest_ind(resampled_pre, resampled_post, equal_var=False)
            bootstrap_pvals.append(pval)
            
            # Compute Cohen's d
            mean_diff = np.mean(resampled_pre) - np.mean(resampled_post)
            pooled_std = np.sqrt((np.std(resampled_pre, ddof=1) ** 2 + np.std(resampled_post, ddof=1) ** 2) / 2)
            cohens_d_val = mean_diff / pooled_std
            bootstrap_cohensd.append(cohens_d_val)

        # Store results
        p_values.append(np.mean(bootstrap_pvals))
        cohens_d.append(np.mean(bootstrap_cohensd))
        cohens_d_ci.append((np.percentile(bootstrap_cohensd, 2.5), np.percentile(bootstrap_cohensd, 97.5)))

    thresholds_arr = np.array(thresholds)
    p_values = np.array(p_values)
    cohens_d = np.array(cohens_d)

    fig = go.Figure()

    # Plot p-values
    fig.add_trace(go.Scatter(
        x=thresholds_arr,
        y=p_values,
        mode="lines+markers",
        name="P-Value",
        line=dict(shape="spline", color="steelblue", width=2),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="<b>Threshold:</b> %{x}<br><b>P-Value:</b> %{y:.3e}<extra></extra>"
    ))

    # Plot Cohen's d with CI as shaded areas
    for i, (d, ci) in enumerate(zip(cohens_d, cohens_d_ci)):
        fig.add_trace(go.Scatter(
            x=[thresholds_arr[i], thresholds_arr[i]],
            y=[ci[0], ci[1]],
            mode="lines",
            line=dict(color="lightcoral", width=2),
            showlegend=False,
            yaxis="y2"
        ))

    # Plot Cohen's d
    fig.add_trace(go.Scatter(
        x=thresholds_arr,
        y=cohens_d,
        mode="lines+markers",
        name="Cohen's d",
        line=dict(shape="spline", color="firebrick", width=2),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>Threshold:</b> %{x}<br><b>Cohen's d:</b> %{y:.2f}<extra></extra>",
        yaxis="y2"
    ))

    # Layout updates
    fig.update_layout(
        title="P-Value and Cohen's d Plot with Confidence Intervals",
        xaxis=dict(title="Threshold Year", tickangle=45, showgrid=True),
        yaxis=dict(title="P-Value", showgrid=True),
        yaxis2=dict(title="Cohen's d", overlaying="y", side="right"),
        template="plotly_white",
        height=600,
        width=900,
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")
    )

    # Show the plot
    fig.show()
    
def plot_violin_for_significant_thresholds(df, significant_thresholds, num_samples=None):
    # Create subplots dynamically for each significant threshold
    num_rows = len(significant_thresholds)
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Threshold: {t}" for t in significant_thresholds]
    )

    for i, threshold in enumerate(significant_thresholds):
        pre_group = df[df["Movie release date"] < threshold]["Score"].dropna()
        post_group = df[df["Movie release date"] >= threshold]["Score"].dropna()

        # Balance groups if num_samples is specified
        if num_samples is not None:
            available_samples = min(len(pre_group), len(post_group), num_samples)
            pre_group = np.random.choice(pre_group, available_samples, replace=False)
            post_group = np.random.choice(post_group, available_samples, replace=False)

        # Add violin plot for pre-group
        fig.add_trace(
            go.Violin(
                y=pre_group,
                name=f"Pre {threshold}",
                box_visible=True,
                meanline_visible=True,
                line_color="steelblue",
                opacity=0.7,
                legendgroup=f"Pre {threshold}",
                hovertemplate="<b>Score:</b> %{y:.2f}<extra></extra>",
            ),
            row=i + 1, col=1
        )

        # Add violin plot for post-group
        fig.add_trace(
            go.Violin(
                y=post_group,
                name=f"Post {threshold}",
                box_visible=True,
                meanline_visible=True,
                line_color="firebrick",
                opacity=0.7,
                legendgroup=f"Post {threshold}",
                hovertemplate="<b>Score:</b> %{y:.2f}<extra></extra>",
            ),
            row=i + 1, col=1
        )

    # Update layout
    fig.update_layout(
        title="Violin Plots for Significant Thresholds",
        height=300 * len(significant_thresholds),  # Dynamic height based on number of significant thresholds
        width=900,
        template="plotly_white",
        showlegend=True,
    )

    # Show the plot
    fig.show()

def run_anova_tukey_test(df, x):
    groups = [group['Score'].values for name, group in df.groupby('Year bin')]
    stat, p_value = f_oneway(*groups)
    print(f"\nANOVA Test Results for bin size = {x}")
    print(f"F-statistic = {stat:.4f}, p-value = {p_value:.6f}")
    
    if p_value < 0.05:
        print("Running Tukey's HSD for pairwise comparisons:")
        
        tukey = pairwise_tukeyhsd(endog=df['Score'], groups=df['Year bin'], alpha=0.05)
        #print(tukey)
        
        tukey.plot_simultaneous()
        plt.title(f"Tukey HSD Pairwise Comparisons (Bin size: {x} years)")
        plt.show()
    else:
        print("\nNo significant differences between bins.")
    return tukey
        
def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / (len(group1) + len(group2) - 2))
    return (mean1 - mean2) / pooled_std

def check_normality(df, threshold):
    pre_group = df[df["Movie release date"] < threshold]["Score"].dropna()
    post_group = df[df["Movie release date"] >= threshold]["Score"].dropna()
    
    print("Normality Check:")
    for group, name in zip([pre_group, post_group], ["Pre-Threshold", "Post-Threshold"]):
        stat, pval = shapiro(group)
        print(f"{name} - Shapiro-Wilk Test p-value: {pval:.5f}")
        if pval < 0.05:
            print(f"  {name} may not be normally distributed (p < 0.05).")
        else:
            print(f"  {name} appears to be normally distributed (p >= 0.05).")
        
        # Visualize distribution
        plt.hist(group, bins=20, alpha=0.6, label=name)
    plt.legend()
    plt.title("Histogram of Scores by Group")
    plt.show()