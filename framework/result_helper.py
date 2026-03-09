import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np


# def paper_tables(df, l, show_process=False):
#     df_filtered = df[df['l'] == l]

#     if show_process:
#         # Group by 'method' and 'process' and calculate mean and std of 'cycle_time'
#         df_summary = df_filtered.groupby(['method', 'process'])['cycle_time'].agg(['mean', 'std']).reset_index().round(2)

#         # Pivot the table to have 'process' as columns
#         return df_summary.pivot(index='method', columns='process', values=['mean', 'std']).transpose()
#     df_summary = df_filtered.groupby('method')['cycle_time'].agg(['mean', 'std']).reset_index().round(2)
#     new_columns = ['method', 'avg_cycle_time', 'std_cycle_time']
#     df_summary.columns = new_columns
#     return df_summary.transpose()

from scipy.stats import t

def paper_tables(df, l, show_process=False, confidence=0.95):
    # Filter the data based on 'l'
    df_filtered = df[df['l'] == l]

    if show_process:
        # Group by 'method' and 'process' to calculate mean, std, and CI
        df_summary = df_filtered.groupby(['method', 'process']).agg(
            mean_cycle_time=('cycle_time', 'mean'),
            std_cycle_time=('cycle_time', 'std'),
            count=('cycle_time', 'count')
        ).reset_index()
        
        # Compute the margin of error for CI
        df_summary['CI'] = df_summary.apply(
            lambda row: t.ppf((1 + confidence) / 2, row['count'] - 1) * (row['std_cycle_time'] / np.sqrt(row['count']))
            if row['count'] > 1 else 0, axis=1
        )

        # Pivot the table to have 'process' as columns
        df_pivot = df_summary.pivot(index='method', columns='process', values=['mean_cycle_time', 'CI']).round(2)
        return df_pivot.transpose()

    # Aggregate statistics for each method
    df_summary = df_filtered.groupby('method').agg(
        avg_cycle_time=('cycle_time', 'mean'),
        std_cycle_time=('cycle_time', 'std'),
        count=('cycle_time', 'count')
    ).reset_index()

    # Compute confidence interval
    df_summary['CI'] = df_summary.apply(
        lambda row: t.ppf((1 + confidence) / 2, row['count'] - 1) * (row['std_cycle_time'] / np.sqrt(row['count']))
        if row['count'] > 1 else 0, axis=1
    )

    # Select relevant columns and round values
    df_summary = df_summary[['method', 'avg_cycle_time', 'CI']].round(2)

    return df_summary.transpose()

def throughput_per_method(df, l, confidence=0.95):
    # Filter the data based on 'l'
    df_filtered = df[df['l'] == l]

    # Aggregate statistics for each method
    df_summary = df_filtered.groupby('method').agg(
        total_count=('count', 'sum'),
        num_runs=('simulation_run', 'nunique')
    ).reset_index()

    # Compute throughput as count per simulation run
    df_summary['throughput'] = df_summary['total_count'] / df_summary['num_runs']

    # Compute standard deviation of throughput per method
    df_std = df_filtered.groupby('method').agg(std_count=('count', 'std')).reset_index()
    df_summary = df_summary.merge(df_std, on='method', how='left')

    # Compute confidence interval
    df_summary['CI'] = df_summary.apply(
        lambda row: t.ppf((1 + confidence) / 2, row['num_runs'] - 1) * (row['std_count'] / np.sqrt(row['num_runs']))
        if row['num_runs'] > 1 else 0, axis=1
    )

    # Select relevant columns and round values
    df_summary = df_summary[['method', 'throughput', 'CI']].round(2)

    return df_summary.transpose()

from scipy.stats import ttest_ind

def compare_methods_ttest(df, method1, method2, alpha=0.05, on='cycle_time'):
    """
    Performs an independent t-test (Welch’s t-test) on the cycle times of two methods.
    
    Parameters:
        df (pd.DataFrame): The dataset containing cycle times.
        method1 (str): Name of the first method.
        method2 (str): Name of the second method.
        alpha (float): Significance level (default is 0.05).
    
    Returns:
        dict: A dictionary with t-statistic, p-value, and significance result.
    """
    # Extract cycle times for the two methods
    cycle_time1 = df[df['method'] == method1][on]
    cycle_time2 = df[df['method'] == method2][on]

    # Perform Welch’s t-test (does not assume equal variances)
    t_stat, p_value = ttest_ind(cycle_time1, cycle_time2, equal_var=False)

    # Determine if the difference is statistically significant
    significant = p_value < alpha

    # Print results
    print(f"T-test results for {method1} vs {method2}:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Statistically significant? {'Yes' if significant else 'No'} (alpha = {alpha})")

    # Return results as a dictionary
    # return {
    #     "method1": method1,
    #     "method2": method2,
    #     "t_stat": round(t_stat, 4),
    #     "p_value": round(p_value, 4),
    #     "significant": significant
    # }

FIGURE_SIZE = (5, 4)
GACT = 'GACT'

def plot_results_broken_y_axis(folder_path, scenario, top=6.4, bottom=18, legend=False, FIGURE_SIZE=(5,4)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import glob
    

    dataframes = []
    csv_files = glob.glob(folder_path)
    for file in csv_files:
        df = pd.read_csv(file, index_col="Unnamed: 0")
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df['status'] == 'COMPLETE']
    
    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")
    #sns.set_theme(context="talk")
    #sns.set_style("darkgrid")

    f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2,
                                          figsize=FIGURE_SIZE, 
                                          sharex=True, 
                                          gridspec_kw={'height_ratios': [1, 6], 'hspace': 0.1})
    
    

    # Create the line plots
    for ax in [ax_top, ax_bottom]:
        sns.lineplot(
            x="l", 
            y="cycle_time",
            hue="method", 
            style="method", 
            data=df,
            hue_order=["Random", "RLRAM","DRL",  "MuProMAC"],
            palette={"RLRAM": sns.color_palette()[1], "DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
            markers={"RLRAM": "o", "DRL": "P", "MuProMAC": "s", "Random": "X"},  
            dashes={"Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2)},
            legend=False,
            ax=ax
        )
        ax.grid(True, linestyle=":", linewidth=1, color=".6")
    
    # Set bold labels
    font_dict = {"fontweight": "bold"}
    ax_bottom.set_xlabel("Lambda (λ)", fontsize=16) # ,**font_dict
    ax_bottom.set_ylabel("GACT", fontsize=16)
    ax_top.set_ylabel("", fontsize=14)
    
    # Set bold ticks
    ax_bottom.tick_params(axis='both', labelsize=13, width=2)
    ax_top.tick_params(axis='both', labelsize=13, width=2)
    for label in ax_bottom.get_xticklabels() + ax_bottom.get_yticklabels():
        label.set_fontweight("bold")
    for label in ax_top.get_xticklabels() + ax_top.get_yticklabels():
        label.set_fontweight("bold")
    
    ax_top.set_ylim(bottom=bottom)
    ax_top.set_xlim(0.18,1.01)
    ax_bottom.set_ylim(bottom=0, top=top)
    ax_bottom.set_xlim(0.18,1.01)
    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_top, bottom=True)
    
    d = .03  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs) 
    kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
    ax_bottom.plot((-d, +d), (1 - d + 0.02, 1 + d - 0.02), **kwargs)  # bottom-left diagonal

    # Tight layout for better spacing
    f.tight_layout()
    f.savefig(f"results/plots/{scenario}_cycle_time_vs_lambda.png", dpi=300, bbox_inches="tight")
    
    # Show the plot
    f.show()

def plot_shared_resources(folder_path, scenario,l=.6, legend=True, fig_name="shared_resources"):
    legend_config = 'auto' if legend else False
    dataframes = []
    csv_files = glob.glob(folder_path)

    for file in csv_files:
        if "papershared2" in file:
            col_val=2
            df = pd.read_csv(file, index_col="Unnamed: 0")
            df['num_shared_r'] = col_val
            dataframes.append(df)
        elif "papershared3" in file:
            col_val=3
            df = pd.read_csv(file, index_col="Unnamed: 0")
            df['num_shared_r'] = col_val
            dataframes.append(df)
        elif "papershared4" in file:
            col_val=4
            df = pd.read_csv(file, index_col="Unnamed: 0")
            df['num_shared_r'] = col_val
            dataframes.append(df)
        elif "papershared5" in file:
            col_val=5
            df = pd.read_csv(file, index_col="Unnamed: 0")
            df['num_shared_r'] = col_val
            dataframes.append(df)
        

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df['status'] == 'COMPLETE']
    df = df[df['method'] != 'Random']
    df_sub = df[df['l']==l]
    #df['process']= df['process'].apply(lambda x: encode_process_names[x])
    #df.rename(columns={'method':'Method', 'process':'Process'}, inplace=True)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")

    # Create the line plot
    plt.figure(figsize=(5,4))
    sns.lineplot(
        x="num_shared_r", 
        y="cycle_time",
        hue="method", 
        style="method", 
        data=df_sub,
        hue_order=["RLRAM","DRL",  "MuProMAC"],
        palette={"RLRAM": sns.color_palette()[1], "DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
        markers={"RLRAM": "o", "DRL": "P", "MuProMAC": "s", "Random": "X"},  
        dashes={"Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2)},
        legend=legend_config
    )
    if legend:
        plt.legend(
            title="Method",
            loc='best',#"best",  # Place legend in the best location
            fontsize=11,
            title_fontsize=14,
            #bbox_to_anchor=(1.8, -0.2),
            ncol = 1
        )
    plt.xlabel("# Shared Resources", fontsize=16)
    plt.ylabel('GACT', fontsize=16)

    # Fine-tune ticks
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Add gridlines and remove top and right spines
    plt.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    sns.despine()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure for publication
    plt.savefig(f"results/plots/{scenario}_{fig_name}.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

def plot_process_details_cycle_time(folder_path, scenario, legend=False, fig_name="cycle_time_vs_lambda_deep_view",encode_process_names={'process_a':'P1', 'process_b':'P2'}):

    legend_config = 'auto' if legend else False
    csv_files = glob.glob(folder_path)
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file,index_col="Unnamed: 0")
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df['status'] == 'COMPLETE']
    df['process']= df['process'].apply(lambda x: encode_process_names[x])
    df.rename(columns={'method':'Method', 'process':'Process'}, inplace=True)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")

    # Create the line plot
    plt.figure(figsize=(5,4))
    sns.lineplot(
        x="l", 
        y="cycle_time",
        hue="Method", 
        style="Process", 
        data=df,
        palette={"RLRAM": sns.color_palette()[1], "DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
        markers={"P1":'o', "P2":'X'}, 
        #dashes={"Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2)}, 
        hue_order=["Random", "RLRAM", "MuProMAC","DRL"],
        legend=legend_config
    )
    if legend:
        plt.legend(
            title="Method & Process",
            loc='lower center',#"best",  # Place legend in the best location
            fontsize=11,
            title_fontsize=14,
            bbox_to_anchor=(1.8, -0.2),
            ncol = 1
        )
    plt.xlabel("Lambda (λ)", fontsize=16)
    plt.ylabel('ACT', fontsize=16)

    # Fine-tune ticks
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Add gridlines and remove top and right spines
    plt.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    sns.despine()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure for publication
    plt.savefig(f"results/plots/{scenario}_{fig_name}.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

def plot_process_details_throughput(folder_path, scenario, legend=False, fig_name="throughput_vs_lambda_deep_view",encode_process_names={'process_a':'p1', 'process_b':'p2'}):

    legend_config = 'auto' if legend else False
    csv_files = glob.glob(folder_path)
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file,index_col="Unnamed: 0")
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df['status'] == 'COMPLETE']
    df['process']= df['process'].apply(lambda x: encode_process_names[x])
    df.rename(columns={'method':'Method', 'process':'Process'}, inplace=True)
    count_data = df.groupby(['l', 'Method', 'Process','simulation_run']).size().reset_index(name='count')
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")

    # Create the line plot
    plt.figure(figsize=(5,4))
    sns.lineplot(
        x="l", 
        y="count",
        hue="Method", 
        style="Process", 
        data=count_data,
        palette={"RLRAM": sns.color_palette()[1], "DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
        markers={"p1":'o', "p2":'X'}, 
        #dashes={"Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2)}, 
        hue_order=["Random", "RLRAM",  "MuProMAC","DRL"],
        legend=legend_config
    )
    if legend:
        plt.legend(
            title="Method & Process",
            loc='lower center',#"best",  # Place legend in the best location
            fontsize=11,
            title_fontsize=14,
            bbox_to_anchor=(1.8, -0.2),
            ncol = 1
        )
    plt.xlabel("Lambda (λ)", fontsize=16)
    plt.ylabel('ATP', fontsize=16)

    # Fine-tune ticks
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Add gridlines and remove top and right spines
    plt.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    sns.despine()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure for publication
    plt.savefig(f"results/plots/{scenario}_{fig_name}.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


def process_data(folder_path):
    csv_files = glob.glob(folder_path)

    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file,index_col="Unnamed: 0")
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df['status'] == 'COMPLETE']
    count_data = df.groupby(['l', 'method', 'process','simulation_run']).size().reset_index(name='count')
    return df, count_data




def plot_results(folder_path, scenario, legend=False):
    csv_files = glob.glob(folder_path)

    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file,index_col="Unnamed: 0")
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df['status'] == 'COMPLETE']
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")

    # Create the line plot
    plt.figure(figsize=FIGURE_SIZE)
    
    sns.lineplot(
        x="l", 
        y="cycle_time",
        hue="method", 
        style="method", 
        data=df,
        markers=True,  # Add markers to emphasize points
        #dashes=False,  # Use solid lines for better readability
        hue_order=["RLRAM","DRL","MuProMAC"],
        legend=legend,
        #alpha=1,
        #linewidth=.3
    )
    
    #Customize the legend
    if legend:
        plt.legend(
            title="Method",
            loc='upper center',#"best",  # Place legend in the best location
            fontsize=11,
            title_fontsize=14,
            bbox_to_anchor=(2, 1)
        )

    # Add titles and labels
    #plt.title("Average Cycle Time vs. Lambda (λ)", fontsize=16, fontweight='bold')
    plt.xlabel("Lambda (λ)", fontsize=14)
    plt.ylabel(GACT, fontsize=14)

    # Fine-tune ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add gridlines and remove top and right spines
    plt.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    #plt.grid(visible=True)
    sns.despine()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure for publication
    plt.savefig(f"results/plots/{scenario}_cycle_time_vs_lambda.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")

    # Create the line plot
    plt.figure(figsize=FIGURE_SIZE)
    sns.lineplot(
        x="l", 
        y="cycle_time",
        hue="method", 
        style="process", 
        data=df,
        markers=True,  # Add markers to emphasize points
        dashes=False,  # Use solid lines for better readability
        hue_order=["RLRAM","DRL","MuProMAC"]
    )

    # Customize the legend
    plt.legend(
        title="Method & Process",
        loc="best",  # Place legend in the best location
        fontsize=11,
        title_fontsize=14,
    )

    # Add titles and labels
    #plt.title("Average Cycle Time vs. Lambda (λ)", fontsize=16, fontweight='bold')
    plt.xlabel("Lambda (λ)", fontsize=14)
    plt.ylabel(GACT, fontsize=14)

    # Fine-tune ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add gridlines and remove top and right spines
    plt.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    sns.despine()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure for publication
    plt.savefig(f"results/plots/{scenario}_cycle_time_vs_lambda.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")

    # Count the occurrences of each combination of l, method, and process
    count_data = df.groupby(['l', 'method', 'process','simulation_run']).size().reset_index(name='count')

    # Create the line plot with counts
    plt.figure(figsize=FIGURE_SIZE)
    sns.lineplot(
        x="l", 
        y="count",  # Use the count column for the y-axis
        hue="method", 
        style="process", 
        data=count_data,
        markers=True,  # Add markers to emphasize points
        dashes=False,  # Use solid lines for better readability
        hue_order=["RLRAM","DRL","MuProMAC"]#["FIFO","RLRAM","DRL","MuProMAC", "Random"]
    )

    # Customize the legend
    plt.legend(
        title="Method & Process",
        loc="best",  # Place legend in the best location
        fontsize=11,
        title_fontsize=14,
    )

    # Add titles and labels
    #plt.title("Count of Cases vs. Lambda (l)", fontsize=16, fontweight='bold')
    plt.xlabel("Lambda (λ)", fontsize=14)
    plt.ylabel("Throughput (TP)", fontsize=14)

    # Fine-tune ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add gridlines and remove top and right spines
    plt.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    sns.despine()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure for publication
    plt.savefig(f"results/plots/{scenario}_throughput_vs_lambda.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()
    return df, count_data
