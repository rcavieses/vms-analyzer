"""
Utility functions for VMS data analysis.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import h3
import folium
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from loguru import logger

def setup_directories(base_path: Path) -> Dict[str, Path]:
    """
    Creates necessary directory structure for VMS analysis.
    
    Args:
        base_path: Base directory path
        
    Returns:
        Dictionary with paths to all necessary directories
    """
    dirs = {
        'input': base_path / 'input',
        'unified': base_path / 'unified',
        'filtered': base_path / 'filtered',
        'classified': base_path / 'classified',
        'analysis': base_path / 'analysis',
        'logs': base_path / 'logs',
        'reports': base_path / 'reports'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def process_single_file(file_path: Path, column_mapping: Dict[str, str], 
                       required_columns: set) -> Optional[pd.DataFrame]:
    """
    Processes a single VMS data file.
    
    Args:
        file_path: Path to the file
        column_mapping: Dictionary mapping original column names to standardized names
        required_columns: Set of required columns
        
    Returns:
        Processed DataFrame or None if processing fails
    """
    try:
        # Read file
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        else:
            df = pd.read_excel(file_path)
        
        # Standardize columns
        df.columns = df.columns.str.strip()
        df.rename(columns=column_mapping, inplace=True)
        
        # Clean data
        df = df.dropna(subset=['Longitud', 'Latitud', 'Fecha'])
        
        # Validate required columns
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in {file_path.name}: {missing_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return None

def process_polygon_chunk(chunk_data: Tuple[pd.DataFrame, Polygon]) -> Optional[gpd.GeoDataFrame]:
    """
    Processes a chunk of data for polygon filtering.
    
    Args:
        chunk_data: Tuple of (DataFrame chunk, Polygon)
        
    Returns:
        Filtered GeoDataFrame or None if processing fails
    """
    chunk, polygon = chunk_data
    try:
        # Create GeoDataFrame
        geometry = gpd.points_from_xy(chunk['Longitud'], chunk['Latitud'])
        points_gdf = gpd.GeoDataFrame(chunk, geometry=geometry, crs="EPSG:4326")
        
        # Filter by polygon
        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        result = gpd.sjoin(points_gdf, polygon_gdf, predicate='within')
        
        return result.drop(columns=['index_right']) if 'index_right' in result.columns else result
        
    except Exception:
        return None

def calculate_hex_times(df: pd.DataFrame, max_gap: float = 2.0) -> pd.DataFrame:
    """
    Calculates fishing time per H3 hexagon.
    
    Args:
        df: DataFrame with VMS data
        max_gap: Maximum time gap (hours) to consider continuous fishing
        
    Returns:
        DataFrame with fishing hours per hexagon
    """
    df = df.sort_values(['Nombre', 'h3_index', 'Fecha'])
    df['time_diff'] = df.groupby(['Nombre', 'h3_index'])['Fecha'].diff().dt.total_seconds() / 3600.0
    
    # Identify continuous periods
    df['new_period'] = (df['time_diff'] > max_gap) | (df['time_diff'].isna())
    df['period_id'] = df.groupby(['Nombre', 'h3_index'])['new_period'].cumsum()
    
    # Calculate time per period
    period_times = df.groupby(['h3_index', 'Nombre', 'period_id']).agg({
        'Fecha': ['min', 'max']
    }).reset_index()
    
    # Calculate duration
    period_times['fishing_hours'] = (
        (period_times[('Fecha', 'max')] - period_times[('Fecha', 'min')])
        .dt.total_seconds() / 3600.0
    )
    
    # Aggregate by hexagon
    hex_times = period_times.groupby('h3_index')['fishing_hours'].sum().reset_index()
    
    return hex_times

def create_hex_gdf(hex_times: pd.DataFrame, resolution: int) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame with H3 hexagon geometries.
    
    Args:
        hex_times: DataFrame with times per hexagon
        resolution: H3 resolution level
        
    Returns:
        GeoDataFrame with hexagon geometries
    """
    geometries = []
    for h3_idx in hex_times['h3_index']:
        vertices = h3.h3_to_geo_boundary(h3_idx, geo_json=True)
        geometries.append(Polygon([(v[1], v[0]) for v in vertices]))
    
    hex_gdf = gpd.GeoDataFrame(
        hex_times,
        geometry=geometries,
        crs="EPSG:4326"
    )
    
    return hex_gdf

def create_effort_map(hex_gdf: gpd.GeoDataFrame,
                     output_path: Path,
                     scale: str = 'linear',
                     title: Optional[str] = None) -> None:
    """
    Creates and saves a fishing effort map.
    
    Args:
        hex_gdf: GeoDataFrame with effort data
        output_path: Path to save the map
        scale: Scale type ('linear' or 'log')
        title: Optional map title
    """
    try:
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Prepare visualization data
        data = hex_gdf.copy()
        column = 'val_millon' if 'val_millon' in data.columns else 'fishing_hours'
        
        if scale == 'log':
            min_non_zero = data[data[column] > 0][column].min()
            data['plot_value'] = data[column].apply(
                lambda x: np.log10(x + min_non_zero/10) if x > 0 else 0
            )
        else:
            data['plot_value'] = data[column]
        
        # Configure visualization
        cmap = plt.cm.YlOrRd
        cmap.set_under('lightgrey')
        
        # Calculate limits
        non_zero_values = data[data['plot_value'] > 0]['plot_value']
        vmin = non_zero_values.min() if len(non_zero_values) > 0 else 0
        vmax = non_zero_values.quantile(0.99) if len(non_zero_values) > 0 else 1
        
        # Create plot
        data.plot(
            column='plot_value',
            cmap=cmap,
            legend=True,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                'label': f'{"Value (Millions $)" if column == "val_millon" else "Fishing Hours"} - {scale} scale',
                'orientation': 'vertical',
                'extend': 'both'
            }
        )
        
        # Add basemap
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            zoom=8
        )
        
        # Customize plot
        if title is None:
            title = f'{"Value" if column == "val_millon" else "Fishing Effort"} Map ({scale} scale)'
        plt.title(title, pad=20, size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Map saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        if plt.get_fignums():
            plt.close()
        raise
def generate_html_report(map_obj: folium.Map, stats: Dict) -> str:
    """
    Generates an HTML report with analysis results.
    
    Args:
        map_obj: Folium map object
        stats: Dictionary with summary statistics
        
    Returns:
        HTML content as string
    """
    # Get map HTML
    map_html = map_obj._repr_html_()
    
    # Create HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Análisis VMS</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }}
            .stats-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .stat-box {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .map-container {{
                width: 100%;
                height: 600px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Reporte de Análisis VMS</h1>
        
        <h2>Estadísticas Generales</h2>
        <div class="stats-container">
            <div class="stat-box">
                <h3>Puntos Totales</h3>
                <p>{stats['total_points']:,}</p>
            </div>
            <div class="stat-box">
                <h3>Puntos Activos</h3>
                <p>{stats['active_points']:,}</p>
            </div>
            <div class="stat-box">
                <h3>Porcentaje de Actividad</h3>
                <p>{stats['activity_percentage']:.1f}%</p>
            </div>
            <div class="stat-box">
                <h3>Hexágonos Únicos</h3>
                <p>{stats['unique_hexagons']:,}</p>
            </div>
            <div class="stat-box">
                <h3>Promedio de Puntos por Hexágono</h3>
                <p>{stats['avg_points_per_hexagon']:.1f}</p>
            </div>
        </div>
        
        <h2>Mapa de Actividad</h2>
        <div class="map-container">
            {map_html}
        </div>
    </body>
    </html>
    """
    
    return html_content