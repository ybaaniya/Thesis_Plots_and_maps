import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec

# Load the data
df_m = pd.read_csv('/Users/yubinbaaniya/Library/CloudStorage/Box-Box/master thesis and what not/Thesis/Analysis files/THAGI saber and jorge.csv')

# Define metrics to plot
metrics = ['me', 'nse', 'rmse', 'r_squared']
suffixes = ['_sim', '_SFDC']

# Define colors for each version
colors = {
    '_sim': '#f4a582',   # coral for _sim version
    '_SFDC': '#9970ab'   # purple for _SFDC version
}

# Define labels for each version
vertical_labels = {
    '_sim': 'Simulated',
    '_SFDC': 'SABER'
}

# Define x-axis limits for each metric
x_limits = {
    'me': (-50, 50),
    'nse': (-1, 1),
    'rmse': (0, 1000),
    'r_squared': (-1, 1)
}

# Create a figure with 4 rows and 2 columns
fig = plt.figure(figsize=(14, 16))
outer_grid = gridspec.GridSpec(4, 2, wspace=0.4, hspace=0.5)

for row_idx, metric in enumerate(metrics):
    for col_idx, suffix in enumerate(suffixes):
        # Define column name
        col = metric + suffix

        # Calculate mean, median, and standard deviation for each metric version
        mean_val = df_m[col].mean()
        median_val = df_m[col].median()
        std_val = df_m[col].std()

        # Define the inner grid for box plot and histogram (within each subplot)
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                      subplot_spec=outer_grid[row_idx, col_idx],
                                                      height_ratios=[1, 4],
                                                      hspace=0.05)

        # Clip data based on x_limits for each metric
        clipped_data = df_m[col].clip(lower=x_limits[metric][0], upper=x_limits[metric][1])

        # Box Plot on top
        ax_box = fig.add_subplot(inner_grid[0])
        sns.boxplot(x=clipped_data, ax=ax_box, showfliers=True, color=colors[suffix])
        ax_box.set(yticks=[], xlabel=None)
        ax_box.set_xlim(x_limits[metric])  # Set x-axis limits from x_limits dictionary

        # Histogram below with specified color and bins divided into 0.25 intervals
        ax_hist = fig.add_subplot(inner_grid[1], sharex=ax_box)
        sns.histplot(clipped_data, binwidth=0.25, binrange=x_limits[metric], ax=ax_hist, color=colors[suffix])
        ax_hist.set_xlabel(metric.upper(), fontsize=12)
        ax_hist.set_ylabel('No. of Gauges')

        # Add mean, median, and standard deviation text in the upper left section of the histogram
        ax_hist.text(x_limits[metric][0] + 0.2, ax_hist.get_ylim()[1] * 0.85,  # Adjust positioning as needed
                     f'μ: {mean_val:.2f}\nM: {median_val:.2f}\nσ: {std_val:.2f}',
                     fontsize=10, color='black', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))


        # Add gridlines to both axes
        ax_box.grid(True, linestyle='--', linewidth=0.5)
        ax_hist.grid(True, linestyle='--', linewidth=0.5)

        # Show the legend for the vertical line only once (on the top-left plot)
        if row_idx == 0 and col_idx == 0:
            ax_hist.legend()

        # Add the vertical label on the right side of the histogram
        ax_hist.text(1.01, 0.5, vertical_labels[suffix], va='center', ha='left', rotation=90,
                     transform=ax_hist.transAxes, fontsize=12, color='black')

# Adjust layout and display the plot
plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.05)
plt.show()