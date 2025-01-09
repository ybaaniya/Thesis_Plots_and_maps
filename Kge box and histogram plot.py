import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec

df_m = pd.read_csv('/Users/yubinbaaniya/Library/CloudStorage/Box-Box/master thesis and what not/Thesis/Analysis files/THAGI saber and jorge.csv')
# Rename columns
df_m.rename(columns={'kge_2012_QM': 'KGE QM', 'kge_sim': 'KGE Sim', 'kge_SFDC': 'KGE SABER'}, inplace=True)

# Define colors for each metric
colors = {
    'KGE QM': '#fed976',
    'KGE Sim': '#f4a582',
    'KGE SABER': '#9970ab'
}

# Define labels for each metric to be displayed vertically
vertical_labels = {
    'KGE Sim': 'Simulated',
    'KGE SABER': 'SABER',
    'KGE QM': 'MFDC-QM'
}

# Create a single figure with three rows (one for each metric)
fig = plt.figure(figsize=(8, 10))

for idx, col in enumerate(['KGE Sim', 'KGE SABER', 'KGE QM']):
    # Calculate mean, median, and standard deviation for each column
    mean_val = df_m[col].mean()
    median_val = df_m[col].median()
    std_val = df_m[col].std()

    # Create a GridSpec for each row with two subplots: one for box plot and one for histogram
    grid = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05, figure=fig)
    grid.update(top=(1 - (idx * 0.3)), bottom=(0.7 - (idx * 0.3)))

    # Clip data so that all values less than -5 are set to -5
    clipped_data = df_m[col].clip(lower=-5)

    # Box Plot on top with specified color
    ax_box = fig.add_subplot(grid[0])
    sns.boxplot(x=clipped_data, ax=ax_box, showfliers=True, color=colors[col])
    ax_box.set(yticks=[], xlabel=None)
    ax_box.set_xlim(-5, 1.05)

    # Histogram below with specified color
    ax_hist = fig.add_subplot(grid[1], sharex=ax_box)
    sns.histplot(clipped_data, binwidth=0.25, binrange=(-5, 1), ax=ax_hist, color=colors[col])
    ax_hist.set_xlabel('KGE',fontsize=12)
    ax_hist.set_ylabel('No. of Gauges')



    # Add mean, median, and standard deviation text in the upper left section of the histogram
    ax_hist.text(-4.8, ax_hist.get_ylim()[1] * 0.85,  # Adjust positioning as needed
                 f'μ: {mean_val:.2f}\nM: {median_val:.2f}\nσ: {std_val:.2f}',
                 fontsize=10, color='black', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))

    # Add a vertical line at -0.41 on the histogram axis
    ax_hist.axvline(-0.41, color='red', linestyle='--', label='KGE: -0.41')

    # Add gridlines to both axes
    ax_box.grid(True, linestyle='--', linewidth=0.5)
    ax_hist.grid(True, linestyle='--', linewidth=0.5)

    # Show the legend for the vertical line only once on the first histogram
    if idx == 0:
        ax_hist.legend()

    # Add the vertical label on the right side of the histogram
    ax_hist.text(1.01, 0.5, vertical_labels[col], va='center', ha='left', rotation=90,
                 transform=ax_hist.transAxes, fontsize=12, color='black')  # Set x closer to 1.01, color to black


# Display the plot
#fig.tight_layout()
plt.subplots_adjust(left=0.1, right=0.96)
#plt.savefig('/Users/yubinbaaniya/Library/CloudStorage/Box-Box/master thesis and what not/Thesis/plot/boxplot and histogram.png', dpi=1200)
plt.show()
