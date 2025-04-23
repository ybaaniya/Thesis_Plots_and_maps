import pandas as pd
import numpy as np
import xarray as xr
import logging
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import hydrostats as hs
from scipy import interpolate
import math
from typing import Tuple

# =============================================================================
# Configuration and Path Setup
# =============================================================================

PATHS = {
    'sfdc_table': Path('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/simulated_monthly_sfdc.zarr'),
    'saber_assign_table': Path(
        '/Users/yubinbaaniya/Documents/geoglows python package/geoglows/data/saber-assign-table_with_coordinate.parquet'),
    'fdc_table': Path('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/simulated_monthly_fdc.zarr'),
    'sim_data': Path('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/2nd_iteration_simulation_data.zarr')
}


# =============================================================================
# Distance Calculation Functions
# =============================================================================

def vincenty_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """
    Calculate the distance and initial bearing between two points on Earth using Vincenty's formulae.
    Args:
        lat1 (float): Latitude of the first point in decimal degrees
        lon1 (float): Longitude of the first point in decimal degrees
        lat2 (float): Latitude of the second point in decimal degrees
        lon2 (float): Longitude of the second point in decimal degrees
    Returns:
        Tuple[float, float]: (distance in meters, initial bearing in degrees)
    """
    # WGS-84 ellipsoidal constants
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening
    b = (1 - f) * a  # semi-minor axis

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lambda1 = math.radians(lon1)
    lambda2 = math.radians(lon2)

    # Reduced latitude (latitude on the auxiliary sphere)
    U1 = math.atan((1 - f) * math.tan(phi1))
    U2 = math.atan((1 - f) * math.tan(phi2))

    # Difference in longitude
    L = lambda2 - lambda1
    lambda_old = L

    # Initialize variables
    sin_sigma = cos_sigma = sigma = sin_alpha = cos_sq_alpha = cos2_sigma_m = 0

    # Iterate until convergence
    for i in range(100):
        sin_sigma = math.sqrt(
            (math.cos(U2) * math.sin(L)) ** 2 +
            (math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(L)) ** 2
        )

        if sin_sigma == 0:
            return 0.0, 0.0  # Coincident points

        cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(U2) * math.cos(L)
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = math.cos(U1) * math.cos(U2) * math.sin(L) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2

        if cos_sq_alpha != 0:
            cos2_sigma_m = cos_sigma - 2 * math.sin(U1) * math.sin(U2) / cos_sq_alpha
        else:
            cos2_sigma_m = 0  # Equatorial line

        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))

        lambda_new = L + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (
                cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)
        )
        )

        if abs(lambda_new - lambda_old) < 1e-12:
            break
        lambda_old = lambda_new
    else:
        raise ValueError("Vincenty formula failed to converge")

    # Calculate distance
    u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    delta_sigma = B * sin_sigma * (
            cos2_sigma_m + B / 4 * (
            cos_sigma * (-1 + 2 * cos2_sigma_m ** 2) -
            B / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma ** 2) *
            (-3 + 4 * cos2_sigma_m ** 2)
    )
    )

    distance = b * A * (sigma - delta_sigma)

    # Calculate initial bearing
    initial_bearing = math.atan2(
        math.cos(U2) * math.sin(L),
        math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(L)
    )
    initial_bearing = math.degrees(initial_bearing)
    initial_bearing = (initial_bearing + 360) % 360

    return distance, initial_bearing


# =============================================================================
# SFDC and FDC Functions
# =============================================================================

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
    """Retrieves data from the SFDC table"""
    ds = xr.open_zarr(PATHS['sfdc_table'])
    filtered_ds = ds.sel(rivid=asgn_mid)
    return filtered_ds.to_dataframe().reset_index()


def retrieve_sfdc_for_river_id(river_id: int or list) -> pd.DataFrame:
    """Retrieves data from the SFDC table using 'asgn_mid' values"""
    asgn_mids = lookup_assign_saber_table(river_id)
    print(f"asgn mid: {asgn_mids}")
    filtered_sfdc = retrieve_sfdc(asgn_mids)
    print(f"filtered sfdc: {filtered_sfdc}")
    return filtered_sfdc


def retrieve_fdc(river_id: int or list) -> pd.DataFrame:
    """Retrieves data from the FDC table"""
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
    """Retrieves simulated data"""
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
    """Performs bias correction"""
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
    """Perform bias correction on ungauged data"""
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


# =============================================================================
# Processing Functions
# =============================================================================

def process_multiple_rivers(input_df: pd.DataFrame,
                            gauge_table_path: str,
                            bias_corrected_dir: str,
                            saber_assign_table_path: str,
                            output_csv_path: str,
                            generate_plots: bool = False,
                            plots_dir: str = None):
    """
    Process multiple river IDs and generate comprehensive metrics.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Create plots directory if needed
    if generate_plots and plots_dir:
        Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Verify input DataFrame has required column
    if 'reach_id' not in input_df.columns:
        raise ValueError("Input DataFrame must contain 'reach_id' column")

    # Read tables once
    gauge_df = pd.read_parquet(gauge_table_path)
    saber_df = pd.read_parquet(saber_assign_table_path)

    # Get river IDs from DataFrame and ensure they're integers
    river_ids = input_df['reach_id'].astype(int).unique()
    logging.info(f"Found {len(river_ids)} unique river IDs to process")

    # Initialize results storage
    results = []

    # Process each river ID with proper tqdm usage
    for river_id in tqdm(river_ids, desc="Processing rivers"):
        try:
            logging.info(f"Processing river ID: {int(river_id)}")

            # Get streamflow data and metrics
            merged_df, kge_metrics, location_info = analyze_streamflow(
                int(river_id),
                gauge_table_path,
                bias_corrected_dir
            )

            # Store comprehensive results
            result_row = {
                'river_id': int(river_id),
                'gauge_id': location_info['gauge_id'],
                'assigned_gauge_id': location_info.get('assigned_gauge_id'),
                'KGE_Qsim': kge_metrics['Qsim'],
                'KGE_Q_SFDC_1st': kge_metrics['Q_SFDC_1st'],
                'KGE_Q_SFDC_2nd': kge_metrics['Q_SFDC_2nd'],
                'vincenty_distance_km': location_info.get('vincenty_distance_km'),
                'bearing_degrees': location_info.get('bearing_degrees'),
                'gauge_latitude': location_info['gauge_lat'],
                'gauge_longitude': location_info['gauge_lon'],
                'saber_latitude': location_info.get('saber_lat'),
                'saber_longitude': location_info.get('saber_lon'),
                'data_points': location_info['data_points'],
                'date_range': location_info['date_range']
            }
            results.append(result_row)

            # Generate plot if requested
            if generate_plots and plots_dir:
                plot_streamflow_comparison(
                    merged_df,
                    kge_metrics,
                    location_info,
                    output_dir=plots_dir
                )

        except Exception as e:
            logging.error(f"Error processing river ID {river_id}: {str(e)}")
            continue

    # Create and save summary DataFrame
    if results:
        summary_df = pd.DataFrame(results)

        # Add some additional statistical summaries
        summary_stats = {
            'mean_distance_km': summary_df['vincenty_distance_km'].mean(),
            'median_distance_km': summary_df['vincenty_distance_km'].median(),
            'max_distance_km': summary_df['vincenty_distance_km'].max(),
            'min_distance_km': summary_df['vincenty_distance_km'].min(),
            'mean_KGE_Q_SFDC_2nd': summary_df['KGE_Q_SFDC_2nd'].mean(),
            'total_stations': len(summary_df)
        }

        # Save detailed results
        summary_df.to_csv(output_csv_path, index=False)

        # Save summary statistics to a separate file
        stats_path = output_csv_path.replace('.csv', '_summary_stats.csv')
        pd.DataFrame([summary_stats]).to_csv(stats_path, index=False)

        logging.info(f"Successfully processed {len(results)} rivers")
        logging.info(f"Results saved to {output_csv_path}")
        logging.info(f"Summary statistics saved to {stats_path}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print("\nKGE Metrics:")
        print(summary_df[['KGE_Qsim', 'KGE_Q_SFDC_1st', 'KGE_Q_SFDC_2nd']].describe())
        print("\nDistance Statistics (km):")
        print(summary_df['vincenty_distance_km'].describe())
    else:
        logging.warning("No results were successfully processed")


def analyze_streamflow(which_river_id: int,
                       gauge_table_path: str,
                       bias_corrected_dir: str,
                       include_vincenty: bool = True) -> tuple:
    """
    Analyzes streamflow data for a given river ID
    """
    try:
        # Get bias correction results (1st iteration)
        logging.info(f"Processing river ID: {which_river_id}")
        bc_results = do_bias_correction_for_me(which_river_id)

        # Read gauge information
        gauge_df = pd.read_parquet(gauge_table_path)
        gauge_info = gauge_df[gauge_df['model_id'] == which_river_id].iloc[0]
        gauge_id = gauge_info['gauge_id']
        model_id = gauge_info['model_id']

        # Read 2nd iteration results
        saber_2nd_path = os.path.join(bias_corrected_dir, f'{model_id}.csv')
        if not os.path.exists(saber_2nd_path):
            raise FileNotFoundError(f"2nd iteration file not found for model_id {model_id}")

        saber_2nd = pd.read_csv(saber_2nd_path)
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

        # Validate data quality
        if merged_df.empty:
            raise ValueError(f"No valid data after merging for river ID {which_river_id}")

        # Calculate flow statistics
        flow_stats = {
            'mean_flow': merged_df['Qobs'].mean(),
            'max_flow': merged_df['Qobs'].max(),
            'min_flow': merged_df['Qobs'].min(),
            'flow_variance': merged_df['Qobs'].var()
        }

        # Calculate KGE metrics
        kge_metrics = {
            'Qsim': hs.kge_2012(merged_df['Qsim'], merged_df['Qobs']),
            'Q_SFDC_1st': hs.kge_2012(merged_df['Q_SFDC_1st'], merged_df['Qobs']),
            'Q_SFDC_2nd': hs.kge_2012(merged_df['Q_SFDC_2nd'], merged_df['Qobs']),
            'Qobs': None
        }

        # Calculate additional performance metrics
        performance_metrics = {
            'nse_Qsim': hs.nse(merged_df['Qsim'], merged_df['Qobs']),
            'nse_Q_SFDC_1st': hs.nse(merged_df['Q_SFDC_1st'], merged_df['Qobs']),
            'nse_Q_SFDC_2nd': hs.nse(merged_df['Q_SFDC_2nd'], merged_df['Qobs'])
        }

        # Get location information
        location_info = {
            'gauge_id': gauge_id,
            'model_id': model_id,
            'gauge_lat': gauge_info['latitude'],
            'gauge_lon': gauge_info['longitude'],
            'data_points': len(merged_df),
            'date_range': f"{merged_df.index.min()} to {merged_df.index.max()}",
            'flow_stats': flow_stats,
            'performance_metrics': performance_metrics
        }

        if include_vincenty:
            # Get SABER coordinates
            saber_df = pd.read_parquet(PATHS['saber_assign_table'])
            saber_info = saber_df[saber_df['model_id'] == which_river_id].iloc[0]

            # Calculate Vincenty distance
            distance, bearing = vincenty_distance(
                gauge_info['latitude'], gauge_info['longitude'],
                saber_info['latitude'], saber_info['longitude']
            )

            location_info.update({
                'saber_lat': saber_info['latitude'],
                'saber_lon': saber_info['longitude'],
                'vincenty_distance_km': distance / 1000,
                'bearing_degrees': bearing,
                'assigned_gauge_id': saber_info['asgn_gid']
            })

        return merged_df, kge_metrics, location_info

    except Exception as e:
        logging.error(f"Error in analyze_streamflow for river ID {which_river_id}: {str(e)}")
        raise


def plot_streamflow_comparison(merged_df: pd.DataFrame,
                               kge_metrics: dict,
                               location_info: dict,
                               output_dir: str = None,
                               plot_type: str = 'both'):
    """
    Creates comprehensive plots comparing different streamflow measurements
    """
    try:
        river_id = location_info['model_id']

        if plot_type in ['timeseries', 'both']:
            # Time series plot
            plt.figure(figsize=(15, 8))
            for column in ['Q_SFDC_1st', 'Q_SFDC_2nd', 'Qsim', 'Qobs']:
                label = f"{column} (KGE={kge_metrics[column]:.2f})" if kge_metrics[column] is not None else column
                plt.plot(merged_df.index, merged_df[column], label=label, alpha=0.7)

            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.ylim(0, merged_df['Qobs'].max() * 1.1)

            # Add comprehensive title with metrics
            title = f'Streamflow Comparison for River {river_id}\n'
            if 'vincenty_distance_km' in location_info:
                title += f'Vincenty Distance: {location_info["vincenty_distance_km"]:.2f} km'

            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f'river_{river_id}_timeseries.png'))
                plt.close()

        if plot_type in ['scatter', 'both']:
            # Scatter plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            plot_data = [
                ('Qsim', 'Original Simulation'),
                ('Q_SFDC_1st', '1st Iteration'),
                ('Q_SFDC_2nd', '2nd Iteration')
            ]

            for ax, (col, title) in zip(axes, plot_data):
                max_val = max(merged_df['Qobs'].max(), merged_df[col].max())
                ax.scatter(merged_df['Qobs'], merged_df[col], alpha=0.5)
                ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
                ax.set_xlabel('Observed Flow (m³/s)')
                ax.set_ylabel('Simulated Flow (m³/s)')
                ax.set_title(f'{title}\nKGE: {kge_metrics[col]:.2f}')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f'river_{river_id}_scatter.png'))
                plt.close()

        # Display plot if not saving
        if not output_dir:
            plt.show()

    except Exception as e:
        logging.error(f"Error in plot_streamflow_comparison for river {river_id}: {str(e)}")
        raise


def main():
    """Main function to run the analysis"""
    # Set up paths and parameters
    input_csv_path = Path(
        '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/tables/bootstrap_metrics_nearest distance and strahler order_with_all necessary metadata.csv')
    gauge_table_path = Path(
        '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/gauge_table_2nd_iteration_deDuplicated cleaned.parquet')
    bias_corrected_dir = Path(
        '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/tables/BIAS Corrected timeseries/Nearest distance and strahler')

    # Read the input CSV file
    df = pd.read_csv(input_csv_path)

    # Filter for specific countries
    countries_of_interest = ['Togo']
    filtered_df = df[df['Country'].isin(countries_of_interest)]

    # Create output path with country information
    countries_str = '_'.join(countries_of_interest).replace(' ', '')
    output_csv_path = Path.home() / 'Downloads' / f"kge_metrics_summaries_{countries_str}.csv"

    print(f"\nProcessing {len(filtered_df)} stations from {len(countries_of_interest)} countries")

    # Process the rivers
    process_multiple_rivers(
        input_df=filtered_df,
        gauge_table_path=str(gauge_table_path),
        bias_corrected_dir=str(bias_corrected_dir),
        saber_assign_table_path=str(PATHS['saber_assign_table']),
        output_csv_path=str(output_csv_path),
        generate_plots=False,
        plots_dir=None  # Specify this directory if you want to save plots
    )