import numpy as np
import pandas as pd
from dtaidistance import dtw
from joblib import Parallel, delayed
# Load the main curves dataframe
main_curves_df = pd.read_csv('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/clusters/cluster_centers_5.csv')
#Ensure that 0 index is maximum value on both the dataframe and is at initial position
main_curves_df = main_curves_df.sort_values(by='Unnamed: 0')
df = main_curves_df.drop(columns=['Unnamed: 0']).sort_index()
#path to zscaled fdc of observed station
observed_data_df = pd.read_csv('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/z_scaled_gauge_2nd_iteration.csv')


# Function to compute the DTW distance for a single row
def compute_dtw_for_row(row):
    row_identifier = row.iloc[0]  # First column value that was disregarded
    observed_row = row.iloc[1:].to_numpy()  # Convert the rest of the row to a numpy array

    # Calculate DTW distance for each column in main_curves_df
    distances = {col: dtw.distance(df[col].to_numpy().flatten(), observed_row) for col in df.columns}

    # Find the column with the minimum DTW distance
    min_distance_column = min(distances, key=distances.get)

    # Return the result for the current row
    return {
        'File': row_identifier,
        'Min_Distance_Column': min_distance_column,
        'Min_Distance_Value': distances[min_distance_column]
    }


# Use joblib to parallelize the computation across multiple cores
results = Parallel(n_jobs=-1)(delayed(compute_dtw_for_row)(row) for index, row in observed_data_df.iterrows())

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_file = '/Users/yubinbaaniya/Downloads/dtw_iteration_2_cluster_5_test.csv'  # 3 column csv with file name, closest cluster center and distance
results_df.to_csv(output_file, index=False)

print(f"Summary file saved to {output_file}")