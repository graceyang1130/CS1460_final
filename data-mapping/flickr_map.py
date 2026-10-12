import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import msgpack
import os
import glob
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import warnings

def load_msgpack_shard_directory(directory_path, max_points=400000):
    """Load geo data from MessagePack shards with lat/lon records."""
    print(f"Loading shards from {directory_path}")
    shard_files = glob.glob(os.path.join(directory_path, '*.msg'))

    if not shard_files:
        print(f"Error: No .msg files found in '{directory_path}'")
        return None

    print(f"Found {len(shard_files)} shards. Loading up to {max_points} points.")
    records = []
    malformed = 0
    valid_points = 0

    pbar = tqdm(total=max_points)
    for i, shard_path in enumerate(shard_files):
        if valid_points >= max_points:
            break
        
        file_name = os.path.basename(shard_path)
        print(f"Processing shard {i+1}/{len(shard_files)}: {file_name}")
        
        try:
            with open(shard_path, 'rb') as f:
                for record in msgpack.Unpacker(f, raw=False):
                    if valid_points >= max_points:
                        break
                        
                    if not isinstance(record, dict):
                        malformed += 1
                        continue
                        
                    lat = record.get('latitude')
                    lon = record.get('longitude')
                    
                    if lat is None or lon is None:
                        malformed += 1
                        continue
                        
                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                        if -90 <= lat_f <= 90 and -180 <= lon_f <= 180:
                            records.append({'latitude': lat_f, 'longitude': lon_f})
                            valid_points += 1
                            pbar.update(1)
                        else:
                            malformed += 1
                    except ValueError:
                        malformed += 1
                        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    pbar.close()

    if malformed:
        print(f"Warning: {malformed} malformed records found")

    if not records:
        print(f"No valid data extracted")
        return None

    df = pd.DataFrame(records)
    original_count = len(df)
    
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90) & 
            (df['longitude'] >= -180) & (df['longitude'] <= 180)]
    
    cleaned_count = len(df)
    if cleaned_count < original_count:
        print(f"Cleaned data: Removed {original_count - cleaned_count} invalid rows")

    if cleaned_count == 0:
        print("No valid geographic data after cleaning")
        return None

    print(f"Loaded {cleaned_count} valid points")
    return df[['latitude', 'longitude']]


def plot_geographic_distribution(df, title, output_name, plot_type='scatter', point_size=1, 
                                alpha=0.6, grid_size=100, kde_levels=10):
    warnings.filterwarnings("ignore", category=UserWarning)
    """Plot geographic data on a world map with various visualization options using GeoPandas."""
    if df is None or df.empty:
        print(f"No data to plot for '{title}'")
        return

    lats = df['latitude'].values
    lons = df['longitude'].values
    num_points = len(lats)
    
    print(f"Plotting {num_points} points as '{plot_type}'")

    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    world_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ne_110m_admin_0_countries.shp")
    
    world = gpd.read_file(world_filepath)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    world.plot(ax=ax, color='lightgray', edgecolor='black')
    
    ax.set_facecolor('aliceblue')
    
    if plot_type == 'scatter':
        if num_points < 2:
            geo_df.plot(ax=ax, markersize=point_size, color='blue', alpha=alpha)
        else:
            try:
                kde = gaussian_kde(np.vstack([lons, lats]))
                densities = kde.evaluate(np.vstack([lons, lats]))
                
                cmap = plt.cm.viridis
                
                scatter = ax.scatter(lons, lats, s=point_size, c=densities, cmap=cmap,
                                    alpha=alpha)
                
                cb = plt.colorbar(scatter, label='Density', orientation='horizontal', 
                                 pad=0.1, shrink=0.6, ax=ax)
                cb.outline.set_visible(False)
            except Exception as e:
                print(f"Density calculation failed: {e}")
                geo_df.plot(ax=ax, markersize=point_size, color='blue', alpha=alpha)
                
    elif plot_type == 'hexbin':
        hb = ax.hexbin(lons, lats, gridsize=grid_size, cmap='cividis', 
                      alpha=alpha, mincnt=1)
        cb = plt.colorbar(hb, label='Density', orientation='horizontal', 
                         pad=0.1, shrink=0.6, ax=ax)
        cb.outline.set_visible(False)
    
    elif plot_type == 'kde':
        try:
            x_min, x_max = -180, 180
            y_min, y_max = -90, 90
            x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
            
            kernel = gaussian_kde(np.vstack([lons, lats]))
            z = np.reshape(kernel(positions).T, x_grid.shape)
            
            contour = ax.contourf(x_grid, y_grid, z, levels=kde_levels, 
                                 cmap='plasma', alpha=alpha)
            cb = plt.colorbar(contour, label='Density', orientation='horizontal', 
                             pad=0.1, shrink=0.6, ax=ax)
            cb.outline.set_visible(False)
        except Exception as e:
            print(f"KDE visualization failed: {e}")
            geo_df.plot(ax=ax, markersize=point_size, color='blue', alpha=alpha)
    else:
        geo_df.plot(ax=ax, markersize=point_size, color='blue', alpha=alpha)

    plt.title(title, fontsize=16)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Format longitude (x-axis) ticks with E/W designations
    lon_ticks = np.arange(-180, 181, 60)
    lon_labels = []
    for lon in lon_ticks:
        if lon == 0:
            lon_labels.append('0°')
        elif lon < 0:
            lon_labels.append(f'{abs(lon)}°W')
        else:
            lon_labels.append(f'{lon}°E')
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels(lon_labels)
    
    # Format latitude (y-axis) ticks with N/S designations
    lat_ticks = np.arange(-90, 91, 30)
    lat_labels = []
    for lat in lat_ticks:
        if lat == 0:
            lat_labels.append('0°')
        elif lat < 0:
            lat_labels.append(f'{abs(lat)}°S')
        else:
            lat_labels.append(f'{lat}°N')
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels(lat_labels)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_name}")
    
    plt.show()

    bbox = (lats.min(), lons.min(), lats.max(), lons.max())
    print(f"Stats: {num_points} points, bbox: {bbox}")


def main():
    print("Geographic Data Distribution Tool")
    print("="*40)

    ## NEED TO CHANGE DIRECTORY OF FLICKR DATA##
    # shards_dir = "../../shards"
    shards_dir = f'/users/hdalvi/scratch/GeoGoat/shards'

    geo_data = load_msgpack_shard_directory(
        directory_path=shards_dir
    )

    if geo_data is not None and not geo_data.empty:
        plot_geographic_distribution(
            df=geo_data, 
            title="Geographic Point Distribution", 
            output_name="out/flickr_spread.png",
            plot_type='scatter',
            point_size=5,
            alpha=0.7
        )
    else:
        print("No data to plot")

    print("Analysis complete")


if __name__ == '__main__':
    main()
