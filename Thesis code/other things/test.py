import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
gauge_data_folder = '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/gauge_data'
model_data_folder = '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/Bias corrected Time series/Archieve/Jorge_method BC'
sim_path = '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/2nd_iteration_simulation_data.zarr'


def retrieve_simulated_data(river_id: int or list, sim_path: str) -> pd.DataFrame:
    """
    Retrieves simulated data for the given river_id(s) from the simulated data file.

    Args:
        river_id (int or list): ID(s) of a stream(s).
        sim_path (str): Path to the Zarr file.

    Returns:
        pd.DataFrame: Filtered DataFrame containing simulated data for the given river_id(s).
    """
    # Open the Zarr file without assuming consolidated metadata
    df = xr.open_zarr(sim_path, consolidated=False)

    # Ensure river_id is compatible with numpy and Python types
    if hasattr(river_id, 'item'):  # Check if it's a numpy type
        river_id = river_id.item()

    if isinstance(river_id, (int, np.integer)):  # Single integer river_id
        simulated_data = df.sel(rivid=river_id)
    elif isinstance(river_id, list):  # List of river_ids
        simulated_data = df.sel(rivid=river_id)
    else:
        raise ValueError("river_id must be an integer or a list of integers")

    # Convert xarray DataArray to a Pandas DataFrame
    simulated_data_df = simulated_data.to_dataframe().reset_index()
    simulated_data_df = simulated_data_df.drop(columns='rivid')  # Drop the 'rivid' column
    simulated_data_df = simulated_data_df.rename(columns={'time': 'date'})  # Rename the 'time' column to 'date'

    return simulated_data_df



def process_and_plot(gauge_id: str, csv_file: str):
    """
    Processes and plots time series data for a specific gauge_id and its corresponding model_id.

    Args:
        gauge_id (str): The ID of the gauge.
        csv_file (str): Path to the main CSV file containing gauge_id and model_id mapping.
    """
    # Load the main CSV file to get model_id
    main_data = pd.read_csv(csv_file)
    main_data = main_data[["gauge_id", "model_id"]]  # Ensure relevant columns
    model_id = main_data.loc[main_data['gauge_id'] == gauge_id, 'model_id'].values[0]

    # Locate corresponding gauge and model CSV files
    gauge_csv_path = os.path.join(gauge_data_folder, f"{gauge_id}.csv")
    model_csv_path = os.path.join(model_data_folder, f"{model_id}.csv")

    # Load time series data
    gauge_data = pd.read_csv(gauge_csv_path)
    model_data = pd.read_csv(model_csv_path)

    # Clean and prepare gauge and model data
    gauge_data = gauge_data.rename(columns={'Datetime': 'date', 'Streamflow (m3/s)': 'flow_gauge'})
    model_data = model_data.rename(columns={'Unnamed: 0': 'date', 'Corrected Simulated Streamflow': 'flow_model'})

    # Debugging: Print column names to verify renaming
    print("Gauge Data Columns:", gauge_data.columns)
    print("Model Data Columns:", model_data.columns)

    # Convert date column to datetime
    gauge_data['date'] = pd.to_datetime(gauge_data['date'])
    model_data['date'] = pd.to_datetime(model_data['date'])

    # Remove NaN and negative values
    gauge_data = gauge_data.dropna().query("flow_gauge >= 0")
    model_data = model_data.dropna().query("flow_model >= 0")

    # Retrieve simulated data from the Zarr file
    simulated_data_df = retrieve_simulated_data(model_id, sim_path)

    # Merge all three datasets on the date column
    combined_data = (
        gauge_data.merge(model_data, on='date', how='inner')
        .merge(simulated_data_df.reset_index(), on='date', how='inner')  # Reset index for merge
    )

    # Debugging: Print the combined data to ensure successful merging
    print("Combined Data Head:")
    print(combined_data.head())

    # Plotting
    plt.figure(figsize=(16, 12))
    plt.plot(combined_data['date'], combined_data['Qout'], label='Simulated')
    plt.plot(combined_data['date'], combined_data['flow_gauge'], label='Observed')
    plt.plot(combined_data['date'], combined_data['flow_model'], label='Bias Corrected')

    plt.xlabel('Date', fontsize=20)  # Increase X-axis label size
    plt.ylabel('Flow (m3/s)', fontsize=20)  # Increase Y-axis label size
    #plt.title(f'Time Series Comparison for Gauge ID {gauge_id} and Model ID {model_id}', fontsize=16)  # Title size
    plt.legend(fontsize=20)
    plt.tick_params(axis='x', labelsize=16)  # Increase size of X-axis tick labels
    plt.tick_params(axis='y', labelsize=16)
    plt.grid(True)
    plt.show()


# Call the function with your specific gauge_id
gauge_id = 'colombia_38017030'  # Replace with your desired gauge_id
csv_file_path = '/Users/yubinbaaniya/Library/CloudStorage/Box-Box/master thesis and what not/Thesis/Analysis files/gauge_table_reduced_1st iteration.csv'  # Path to main CSV file
process_and_plot(gauge_id, csv_file_path)
