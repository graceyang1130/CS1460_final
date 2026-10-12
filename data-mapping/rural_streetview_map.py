import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.stats import gaussian_kde
import os

from flickr_map import plot_geographic_distribution

def load_coords_csv(filepath, dataset_name=None):
    """Load geographic coordinates from CSV file with flexible header handling."""
    if dataset_name is None:
        dataset_name = os.path.basename(filepath)
        
    print(f"Loading {dataset_name} from {filepath}")
    
    try:
        # Read the file
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")

        # Handle column names
        if 'latitude' in df.columns and 'longitude' in df.columns:
            pass  # Standard column names found
        elif len(df.columns) == 2:
            print(f"Assuming columns: {df.columns[0]}=latitude, {df.columns[1]}=longitude")
            df.columns = ['latitude', 'longitude']
        else:
            print(f"Error: CSV must have 'latitude' and 'longitude' columns, or exactly two columns")
            print(f"Found: {df.columns.tolist()}")
            return None

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Clean the data
    original_count = len(df)
    
    # Convert to numeric and filter valid ranges
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90) & 
            (df['longitude'] >= -180) & (df['longitude'] <= 180)]
    
    cleaned_count = len(df)
    if cleaned_count < original_count:
        print(f"Cleaned data: Removed {original_count - cleaned_count} invalid rows")

    if cleaned_count == 0:
        print("No valid coordinates found after cleaning")
        return None

    print(f"Successfully loaded {cleaned_count} valid coordinates")
    return df[['latitude', 'longitude']]


def main():
    print("CSV Coordinate Plotting Tool")
    print("="*30)

    # File configuration
    csv_filepath = "./coords.csv"
    
    # Load and plot data
    coords_df = load_coords_csv(filepath=csv_filepath)

    if coords_df is not None and not coords_df.empty:
        # Plot with density coloring
        plot_geographic_distribution(
            df=coords_df,
            title=f"Geographic Distribution",
            output_name="streeview_spread.png",
            plot_type='scatter',
            point_size=10,
            alpha=0.7
        )
    else:
        print(f"Could not plot data from {csv_filepath}")

    print("Done")


if __name__ == '__main__':
    main()