import pandas as pd
import numpy as np
import xarray as xr
from scipy import interpolate
from pathlib import Path
from distance import vincenty_distance
import hydrostats as hs
import matplotlib.pyplot as plt

# Import paths from main
from main import PATHS


def make_interpolator(x: np.array, y: np.array, extrap: str = 'nearest',
                      fill_value: int or float = None) -> interpolate.interp1d:
    """
    Make an interpolator from two arrays
    Args:
        x: x values
        y: y values
        extrap: method for extrapolation: nearest, const, linear, average, max, min
        fill_value: value to use when extrap='const'
    Returns:
        interpolate.interp1d
    """
    if extrap == 'nearest':
        return interpolate.interp1d(x, y, fill_value='extrapolate', kind='nearest')
    elif extrap == 'const':
        if fill_value is None:
            raise ValueError('Must provide the const kwarg when extrap_method="const"')
        return interpolate.interp1d(x, y, fill_value=fill_value, bounds_error=False)
    elif extrap == 'linear':
        return interpolate.interp1d(x, y, fill_value='extrapolate')
    elif extrap == 'average':
        return interpolate.interp1d(x, y, fill_value=np.mean(y), bounds_error=False)
    elif extrap in ['max', 'maximum']:
        return interpolate.interp1d(x, y, fill_value=np.max(y), bounds_error=False)
    elif extrap in ['min', 'minimum']:
        return interpolate.interp1d(x, y, fill_value=np.min(y), bounds_error=False)
    else:
        raise ValueError('Invalid extrapolation method provided')


def lookup_assign_saber_table(river_id: int or list) -> list:
    """
    Retrieves 'asgn_mid' values from the SABER assign table
    Args:
        river_id: ID(s) of a stream(s), should be a 9-digit integer or a list of such integers
    Returns:
        list: List of 'asgn_mid' values
    """
    if hasattr(river_id, 'item'):
        river_id = river_id.item()
    if isinstance(river_id, (int, np.integer)):
        river_id = [river_id]
    elif isinstance(river_id, list):
        if not all(isinstance(x, int) for x in river_id):
            raise ValueError("All river_id values must be integers")
    else:
        raise ValueError("river_id must be an integer or a list of integers")

    df = pd.read_parquet(PATHS['saber_assign_table'])
    asgn_mid = df.loc[df['model_id'].isin(river_id), 'asgn_mid'].tolist()

    # Additional info
    gauge_id = df.loc[df['model_id'].isin(river_id), 'asgn_gid'].tolist()
    latitude = df.loc[df['model_id'].isin(river_id), 'latitude'].tolist()
    longitude = df.loc[df['model_id'].isin(river_id), 'longitude'].tolist()

    print(f"asgn_gid in 1st iteration: {gauge_id}")
    print(f"latitude in 1st iteration: {latitude}")
    print(f"longitude in 1st iteration: {longitude}")

    return asgn_mid


def retrieve_sfdc(asgn_mid: int) -> pd.DataFrame:
    """
    Retrieves data from the SFDC table
    Args:
        asgn_mid: Assignment ID to filter the data
    Returns:
        pd.DataFrame: Filtered data containing only rows where 'rivid' matches 'asgn_mid'
    """
    ds = xr.open_zarr(PATHS['sfdc_table'])
    filtered_ds = ds.sel(rivid=asgn_mid)
    return filtered_ds.to_dataframe().reset_index()


def retrieve_sfdc_for_river_id(river_id: int or list) -> pd.DataFrame:
    """
    Retrieves data from the SFDC table using 'asgn_mid' values
    Args:
        river_id: ID(s) of a stream(s)
    Returns:
        pd.DataFrame: Filtered DataFrame from the SFDC table
    """
    asgn_mids = lookup_assign_saber_table(river_id)
    print(f"asgn mid: {asgn_mids}")
    filtered_sfdc = retrieve_sfdc(asgn_mids)
    print(f"filtered sfdc: {filtered_sfdc}")
    return filtered_sfdc


def retrieve_fdc(river_id: int or list) -> pd.DataFrame:
    """
    Retrieves data from the FDC table
    Args:
        river_id: ID(s) of a stream(s)
    Returns:
        pd.DataFrame: Filtered DataFrame from the FDC table
    """
    if hasattr(river_id, 'item'):
        river_id = river_id.item()
    if isinstance(river_id, (int, np.integer)):
        river_id = [river_id]
    elif isinstance(river_id, list):
        if not all(isinstance(x, int) for x in river_id):
            raise ValueError("All river_id values must be integers")
    else:
        raise ValueError("river_id must be an integer or a list of integers")

    df = xr.open_zarr(PATHS['fdc_table'])
    fdc = df.sel(rivid=river_id)
    return fdc.to_dataframe().reset_index()


def retrieve_simulated_data(river_id: int or list) -> pd.DataFrame:
    """
    Retrieves simulated data
    Args:
        river_id: ID(s) of a stream(s)
    Returns:
        pd.DataFrame: Filtered DataFrame containing simulated data
    """
    df = xr.open_zarr(PATHS['sim_data'])
    if hasattr(river_id, 'item'):
        river_id = river_id.item()
    if isinstance(river_id, (int, np.integer)):
        simulated_data = df.sel(rivid=river_id)
    elif isinstance(river_id, list):
        simulated_data = df.sel(rivid=river_id)
    else:
        raise ValueError("river_id must be an integer or a list of integers")

    simulated_data_df = simulated_data.to_dataframe().reset_index()
    return simulated_data_df.set_index('time')


def do_bias_correction_for_me(river_id: int) -> pd.DataFrame:
    """
    Performs bias correction for a given river ID
    Args:
        river_id: ID of the stream
    Returns:
        pd.DataFrame: DataFrame containing bias-corrected data
    """
    simulated_data = retrieve_simulated_data(river_id=river_id)
    sfdc_b = retrieve_sfdc_for_river_id(river_id=river_id)
    sim_fdc_b = retrieve_fdc(river_id=river_id)

    monthly_results = []

    for month in sorted(set(simulated_data.index.month)):
        mon_sim_b = simulated_data[simulated_data.index.month == month].dropna().clip(lower=0)
        qb_original = mon_sim_b['Qout'].values.flatten()

        scalar_fdc = sfdc_b[sfdc_b['month'] == month][['p_exceed', 'sfdc']].set_index('p_exceed')
        sim_fdc_b_m = sim_fdc_b[sim_fdc_b['month'] == month][['p_exceed', 'fdc']].set_index('p_exceed')

        flow_to_percent = make_interpolator(
            sim_fdc_b_m.values.flatten(),
            sim_fdc_b_m.index,
            extrap='nearest',
            fill_value=None
        )

        percent_to_scalar = make_interpolator(
            scalar_fdc.index,
            scalar_fdc.values.flatten(),
            extrap='nearest',
            fill_value=None
        )

        p_exceed = flow_to_percent(qb_original)
        scalars = percent_to_scalar(p_exceed)
        qb_adjusted = qb_original / scalars

        month_df = pd.DataFrame({
            'date': mon_sim_b.index,
            'Simulated': qb_original,
            'Bias Corrected Simulation': qb_adjusted
        })

        monthly_results.append(month_df)

    return pd.concat(monthly_results).set_index('date').sort_index()


def bias_correct_ungauge(simulated_data: pd.DataFrame, sfdc: pd.DataFrame) -> pd.DataFrame:
    """
    Perform bias correction on the simulated data using the SFDC table
    Args:
        simulated_data: DataFrame containing the simulated data to be bias-corrected
        sfdc: DataFrame containing the SFDC table for the corresponding river
    Returns:
        pd.DataFrame: DataFrame with bias-corrected simulated data
    """
    monthly_results = []

    for month in sorted(set(simulated_data.index.month)):
        mon_sim = simulated_data[simulated_data.index.month == month].dropna().clip(lower=0)
        qb_original = mon_sim['Qout'].values.flatten()

        scalar_fdc = sfdc[sfdc['month'] == month][['p_exceed', 'sfdc']].set_index('p_exceed')

        flow_to_percent = make_interpolator(
            scalar_fdc['sfdc'].values.flatten(),
            scalar_fdc.index,
            extrap='nearest',
            fill_value=None
        )

        p_exceed = flow_to_percent(qb_original)

        percent_to_scalar = make_interpolator(
            scalar_fdc.index,
            scalar_fdc['sfdc'].values.flatten(),
            extrap='nearest',
            fill_value=None
        )

        scalars = percent_to_scalar(p_exceed)
        qb_adjusted = qb_original / scalars

        month_df = pd.DataFrame({
            'date': mon_sim.index,
            'Simulated': qb_original,
            'Bias Corrected Simulation': qb_adjusted
        })

        monthly_results.append(month_df)

    return pd.concat(monthly_results).set_index('date').sort_index()


def analyze_streamflow(which_river_id: int, gauge_table_path: str, bias_corrected_dir: str) -> tuple:
    """
    Analyzes streamflow data and calculates Vincenty distance between gauge and SABER coordinates.
    """
    bc_results = do_bias_correction_for_me(which_river_id)
    # Read gauge information once
    gauge_df = pd.read_parquet(gauge_table_path)
    gauge_info = gauge_df[gauge_df['model_id'] == which_river_id].iloc[0]
    gauge_id = gauge_info['gauge_id']
    model_id = gauge_info['model_id']

    # Extract gauge coordinates
    gauge_lat = gauge_info['latitude']
    gauge_lon = gauge_info['longitude']

    # Get SABER coordinates
    saber_df = pd.read_parquet(saber_assign_table_path)
    saber_info = saber_df[saber_df['model_id'] == which_river_id].iloc[0]
    saber_lat = saber_info['latitude']
    saber_lon = saber_info['longitude']

    # Calculate Vincenty distance and bearing
    distance, bearing = vincenty_distance(
        gauge_lat, gauge_lon,
        saber_lat, saber_lon
    )

    # Print coordinate information with Vincenty distance
    print("\nCoordinate Information:")
    print(f"Gauge ID: {gauge_id}")
    print(f"Model ID: {model_id}")
    print(f"Gauge Coordinates: ({gauge_lat:.4f}, {gauge_lon:.4f})")
    print(f"SABER Coordinates: ({saber_lat:.4f}, {saber_lon:.4f})")
    print(f"Vincenty Distance: {distance / 1000:.2f} kilometers")
    print(f"Bearing: {bearing:.2f} degrees")
    print("-" * 50)

    # Read 2nd iteration results
    saber_2nd = pd.read_csv(f'{bias_corrected_dir}/{model_id}.csv')

    # Process 2nd iteration data
    saber_2nd = (saber_2nd
                 .rename(columns={'Unnamed: 0': 'date', 'Qmod': 'Q_SFDC_2nd'})
                 .assign(date=lambda x: pd.to_datetime(x['date']))
                 .set_index('date'))

    # Merge datasets efficiently
    merged_df = (pd.merge(bc_results, saber_2nd,
                          left_index=True,
                          right_index=True,
                          how='inner')
                 .drop(columns='Simulated')
                 .rename(columns={'Bias Corrected Simulation': 'Q_SFDC_1st'})
                 .replace([np.inf, -np.inf], np.nan)
                 .assign(Qobs=lambda x: x['Qobs'].clip(lower=0))
                 .dropna())

    # Calculate KGE metrics once
    kge_metrics = {
        'Qsim': hs.kge_2012(merged_df['Qsim'], merged_df['Qobs']),
        'Q_SFDC_1st': hs.kge_2012(merged_df['Q_SFDC_1st'], merged_df['Qobs']),
        'Q_SFDC_2nd': hs.kge_2012(merged_df['Q_SFDC_2nd'], merged_df['Qobs']),
        'Qobs': None
    }

    return merged_df, kge_metrics


# Keep the original plotting function unchanged
def plot_streamflow_comparison(merged_df: pd.DataFrame, kge_metrics: dict, year: int = None):
    """
    Creates a plot comparing different streamflow measurements with KGE metrics.

    Args:
        merged_df (pd.DataFrame): DataFrame containing streamflow data
        kge_metrics (dict): Dictionary containing KGE metrics for each measurement
        year (int, optional): Year to filter the data, if provided.
    """
    plt.figure(figsize=(15, 8))

    if year:
        merged_df = merged_df.loc[merged_df.index.year == year]

    for column in ['Q_SFDC_1st', 'Q_SFDC_2nd', 'Qsim', 'Qobs']:
        label = f"{column} (KGE={kge_metrics[column]:.2f})" if kge_metrics[column] is not None else column
        plt.plot(merged_df.index, merged_df[column], label=label)

    plt.xlabel('Date')
    plt.ylabel('Streamflow (m³/s)')
    plt.ylim(0)
    plt.title(
        f'Streamflow Comparison for {which_river_id} (Year {year})' if year else f'Streamflow Comparison for {which_river_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
