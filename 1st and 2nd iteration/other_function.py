import pandas as pd
import logging
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import hydrostats as hs
from distance import vincenty_distance
from extract_sfdc import do_bias_correction_for_me
from distance import analyze_streamflow


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

    # Get river IDs from DataFrame
    river_ids = input_df['reach_id'].unique()
    logging.info(f"Found {len(river_ids)} unique river IDs to process")

    # Initialize results storage
    results = []

    # Process each river ID
    for river_id in tqdm.tqdm(river_ids, desc="Processing rivers"):
        try:
            logging.info(f"Processing river ID: {river_id}")
            # Get streamflow data and metrics
            merged_df, kge_metrics, location_info = analyze_streamflow(
                river_id,
                gauge_table_path,
                bias_corrected_dir
            )

            # Store comprehensive results
            result_row = {
                'river_id': river_id,
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
        summary_df.to_csv(output_csv_path, index=False)
        logging.info(f"Results saved to {output_csv_path}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print("\nKGE Metrics:")
        print(summary_df[['KGE_Qsim', 'KGE_Q_SFDC_1st', 'KGE_Q_SFDC_2nd']].describe())
        print("\nDistance Statistics (km):")
        print(summary_df['vincenty_distance_km'].describe())
    else:
        logging.warning("No results were successfully processed")

# The rest of your functions remain the same