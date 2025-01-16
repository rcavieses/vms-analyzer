"""
Main module for VMS data analysis.
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import multiprocessing as mp
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from loguru import logger
import h3
import folium
from tqdm import tqdm

from . import utils

class VMSAnalyzer:
    """
    Unified library for VMS (Vessel Monitoring System) data analysis.
    Handles the complete workflow from file unification to visualization
    and spatial analysis.
    """
    
    def __init__(self, 
                 base_path: Path,
                 chunk_size: int = 5_000_000,
                 n_cores: Optional[int] = None):
        """
        Initialize the VMS analyzer.
        
        Args:
            base_path: Base directory for input/output
            chunk_size: Chunk size for processing
            n_cores: Number of cores for parallel processing
        """
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.n_cores = n_cores if n_cores is not None else mp.cpu_count() - 1
        
        # Set up directories
        self.dirs = utils.setup_directories(self.base_path)
        
        # Configure logging
        log_path = self.dirs['logs'] / f"vms_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.remove()
        logger.add(log_path, rotation="100 MB")
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
        
        # Standard column mapping
        self.column_mapping = {
            'Embarcación': 'Nombre',
            'Nombre_Embarcación': 'Nombre',
            'Descripcion': 'Puerto Base',
            'Permisionario o Concesionario': 'Razón Social',
            'Descripcion.1': 'Razón Social',
            'Razón_Social': 'Razón Social',
            'PERMISIONARIO O CONCESIONARIO': 'Razón Social',
            'FechaRecepcionUnitrac': 'Fecha',
            'Permisionario o consesionario': 'Razón Social'
        }
        
        # Required columns
        self.required_columns = {
            'Nombre', 'Razón Social', 'Fecha', 'Longitud', 
            'Latitud', 'Velocidad'
        }

    def unify_files(self, input_folder: Optional[Path] = None) -> Path:
        """
        Unifies multiple VMS files into one, standardizing columns.
        
        Args:
            input_folder: Folder with VMS files (optional)
            
        Returns:
            Path to unified file
        """
        input_folder = input_folder or self.dirs['input']
        output_file = self.dirs['unified'] / 'vms_unified.csv'
        
        try:
            # Get file list
            files = []
            for ext in ['.csv', '.xlsx']:
                files.extend(input_folder.glob(f'*{ext}'))
            
            if not files:
                raise ValueError(f"No files found in {input_folder}")
            
            logger.info(f"Processing {len(files)} files...")
            
            # Process first file
            first_df = utils.process_single_file(files[0], 
                                               self.column_mapping, 
                                               self.required_columns)
            first_df.to_csv(output_file, index=False, encoding='utf-8')
            total_rows = len(first_df)
            
            # Process remaining files
            for file_path in tqdm(files[1:], desc="Unifying files"):
                df = utils.process_single_file(file_path, 
                                            self.column_mapping, 
                                            self.required_columns)
                if df is not None:
                    df.to_csv(output_file, mode='a', header=False, 
                             index=False, encoding='utf-8')
                    total_rows += len(df)
            
            logger.info(f"Unification completed: {total_rows:,} total records")
            return output_file
            
        except Exception as e:
            logger.error(f"Error in unification: {e}")
            raise

    def filter_by_polygon(self, 
                         input_file: Optional[Path] = None,
                         vertices_file: Optional[Path] = None) -> Path:
        """
        Filters VMS data by a defined polygon.
        
        Args:
            input_file: VMS file to filter (optional)
            vertices_file: File with polygon vertices (optional)
            
        Returns:
            Path to filtered file
        """
        input_file = input_file or self.dirs['unified'] / 'vms_unified.csv'
        vertices_file = vertices_file or self.dirs['input'] / 'vertices.csv'
        output_file = self.dirs['filtered'] / 'vms_filtered.csv'
        
        try:
            # Load polygon
            vertices_df = pd.read_csv(vertices_file)
            vertices = [(row['Longitud'], row['Latitud']) 
                       for _, row in vertices_df.iterrows()]
            if vertices[0] != vertices[-1]:
                vertices.append(vertices[0])
            polygon = Polygon(vertices)
            
            # Set up parallel processing
            pool = mp.Pool(self.n_cores)
            
            # Process in chunks
            # Process in chunks
            filtered_chunks = []
            with tqdm(desc="Filtering by polygon") as pbar:
                for chunk in pd.read_csv(input_file, chunksize=self.chunk_size):
                    # Create points for each coordinate pair
                    points = [
                        (row['Longitud'], row['Latitud']) 
                        for _, row in chunk.iterrows()
                    ]
                    
                    # Parallel check if points are within polygon
                    results = pool.map(
                        lambda p: polygon.contains(Point(p)), 
                        points
                    )
                    
                    # Filter chunk based on results
                    filtered_chunk = chunk[results]
                    if not filtered_chunk.empty:
                        filtered_chunks.append(filtered_chunk)
                    
                    pbar.update(len(chunk))
            
            pool.close()
            pool.join()
            
            # Combine filtered chunks and save
            if filtered_chunks:
                pd.concat(filtered_chunks).to_csv(
                    output_file, 
                    index=False, 
                    encoding='utf-8'
                )
                logger.info(f"Filtering completed: {sum(len(c) for c in filtered_chunks):,} records within polygon")
            else:
                logger.warning("No points found within polygon")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error in polygon filtering: {e}")
            raise

    def analyze_activity(self, 
                        input_file: Optional[Path] = None,
                        resolution: int = 8,
                        speed_threshold: float = 2.0,
                        analysis_type: str = 'density') -> Tuple[gpd.GeoDataFrame, Dict]:
        """
        Analyzes vessel activity patterns using H3 hexagons.
        
        Args:
            input_file: VMS file to analyze (optional)
            resolution: H3 hexagon resolution (default: 8)
            speed_threshold: Speed threshold for activity classification (default: 2.0)
            analysis_type: Type of analysis to perform ('density' or 'hours') (default: 'density')
                - 'density': Analyzes density of points and active points per hexagon
                - 'hours': Calculates total fishing hours per hexagon
            
        Returns:
            Tuple containing:
            - GeoDataFrame with hexagon geometries and metrics
            - Dictionary with summary statistics
        """
        input_file = input_file or self.dirs['filtered'] / 'vms_filtered.csv'
        
        try:
            logger.info(f"Starting activity analysis using {analysis_type} method...")
            
            if analysis_type not in ['density', 'hours']:
                raise ValueError("analysis_type must be either 'density' or 'hours'")
            
            # Define datatypes for columns
            dtypes = {
                'Nombre': str,
                'Razón Social': str,
                'Puerto Base': str,
                'Fecha': str,
                'Latitud': float,
                'Longitud': float,
                'Velocidad': float
            }
            
            if analysis_type == 'density':
                # Process data in chunks for density analysis
                hex_counts = {}
                total_points = 0
                active_points = 0
                
                for chunk in tqdm(pd.read_csv(input_file, 
                                            chunksize=self.chunk_size,
                                            dtype=dtypes,
                                            low_memory=False),
                                desc="Processing chunks"):
                    # Convert coordinates to H3 indexes
                    h3_indexes = [
                        h3.latlng_to_cell(lat, lon, resolution)
                        for lat, lon in zip(chunk['Latitud'], chunk['Longitud'])
                    ]
                    
                    # Count points and active points per hexagon
                    for idx, speed in zip(h3_indexes, chunk['Velocidad']):
                        if idx not in hex_counts:
                            hex_counts[idx] = {'total': 0, 'active': 0}
                        
                        hex_counts[idx]['total'] += 1
                        if speed <= speed_threshold:
                            hex_counts[idx]['active'] += 1
                            active_points += 1
                        
                    total_points += len(chunk)
                
                # Create hexagon geometries for density analysis
                hexagons = []
                for h3_idx, counts in hex_counts.items():
                    boundaries = h3.cell_to_boundary(h3_idx)
                    poly = Polygon(boundaries)
                    
                    activity_pct = (counts['active'] / counts['total']) * 100
                    
                    hexagons.append({
                        'geometry': poly,
                        'h3_index': h3_idx,
                        'total_points': counts['total'],
                        'active_points': counts['active'],
                        'activity_percentage': activity_pct
                    })
                
                # Calculate density analysis statistics
                stats = {
                    'total_points': total_points,
                    'active_points': active_points,
                    'activity_percentage': (active_points / total_points) * 100,
                    'unique_hexagons': len(hex_counts),
                    'avg_points_per_hexagon': total_points / len(hex_counts)
                }
                
            else:  # analysis_type == 'hours'
                # Leer todos los datos
                df = pd.concat(
                    [chunk for chunk in tqdm(
                        pd.read_csv(input_file, 
                                chunksize=self.chunk_size,
                                dtype=dtypes,
                                low_memory=False),
                        desc="Reading data"
                    )],
                    ignore_index=True
                )
                
                # Filtrar por velocidad
                df = df[df['Velocidad'] <= speed_threshold].copy()
                
                # Convertir fechas
                df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
                
                # Crear GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df['Longitud'], df['Latitud']),
                    crs='EPSG:4326'
                )
                
                # Asignar índices H3
                logger.info("Calculating H3 indexes...")
                gdf['h3_index'] = gdf.apply(
                    lambda row: h3.latlng_to_cell(
                        row.geometry.y,
                        row.geometry.x,
                        resolution
                    ),
                    axis=1
                )
                
                # Calcular períodos de pesca
                logger.info("Calculating fishing periods...")
                gdf = gdf.sort_values(['Nombre', 'h3_index', 'Fecha'])
                gdf['time_diff'] = gdf.groupby(['Nombre', 'h3_index'])['Fecha'].diff().dt.total_seconds() / 3600.0
                
                # Identificar períodos continuos
                MAX_GAP = 2  # horas
                gdf['new_period'] = (gdf['time_diff'] > MAX_GAP) | (gdf['time_diff'].isna())
                gdf['period_id'] = gdf.groupby(['Nombre', 'h3_index'])['new_period'].cumsum()
                
                # Calcular tiempo por período
                logger.info("Calculating time per period...")
                period_times = gdf.groupby(['h3_index', 'Nombre', 'period_id']).agg({
                    'Fecha': ['min', 'max']
                }).reset_index()
                
                period_times['fishing_hours'] = (
                    (period_times[('Fecha', 'max')] - period_times[('Fecha', 'min')])
                    .dt.total_seconds() / 3600.0
                )
                
                # Agregar por hexágono
                hex_times = period_times.groupby('h3_index')['fishing_hours'].sum().reset_index()
                
                # Crear GeoDataFrame final
                logger.info("Creating final GeoDataFrame...")
                hexagons = []
                total_hours = 0
                
                for _, row in hex_times.iterrows():
                    boundaries = h3.cell_to_boundary(row['h3_index'])
                    poly = Polygon([(vertex[1], vertex[0]) for vertex in boundaries])
                    
                    hexagons.append({
                        'geometry': poly,
                        'h3_index': row['h3_index'],
                        'fishing_hours': row['fishing_hours']
                    })
                    total_hours += row['fishing_hours']
                
                stats = {
                    'total_fishing_hours': total_hours,
                    'unique_hexagons': len(hexagons),
                    'avg_hours_per_hexagon': total_hours / len(hexagons) if hexagons else 0,
                    'max_hours_in_hexagon': max(h['fishing_hours'] for h in hexagons) if hexagons else 0
                }
            
            # Crear GeoDataFrame final
            gdf = gpd.GeoDataFrame(hexagons)
            if not gdf.empty and not gdf.crs:
                gdf.set_crs(epsg=4326, inplace=True)
            
            logger.info(f"Activity analysis completed using {analysis_type} method")
            logger.info("Estadísticas de Actividad Pesquera:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{key}: {value:.2f}")
                else:
                    logger.info(f"{key}: {value}")
            
            return gdf, stats
            
        except Exception as e:
            logger.error(f"Error in activity analysis: {e}")
            raise
    
    def generate_report(self, 
                       gdf: gpd.GeoDataFrame, 
                       stats: Dict,
                       analysis_type: str = 'density',
                       output_file: Optional[Path] = None) -> Path:
        """
        Generates an HTML report with analysis results.
        
        Args:
            gdf: GeoDataFrame with analysis results
            stats: Dictionary with summary statistics
            analysis_type: Type of analysis ('density' or 'hours')
            output_file: Path for output HTML file (optional)
            
        Returns:
            Path to generated report
        """
        # Asegurar que existe el directorio reports
        if 'reports' not in self.dirs:
            self.dirs['reports'] = self.dirs['analysis'] / 'reports'
            self.dirs['reports'].mkdir(exist_ok=True)
        
        output_file = output_file or self.dirs['reports'] / f'vms_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        try:
            # Crear directorio temporal para archivos estáticos
            temp_dir = self.dirs['reports'] / 'temp'
            temp_dir.mkdir(exist_ok=True)
            
            # Generar mapa estático para el reporte
            map_file = temp_dir / 'activity_map.png'
            utils.create_effort_map(
                hex_gdf=gdf,
                output_path=map_file,
                scale='linear',
                title=f'{"Densidad de Actividad" if analysis_type == "density" else "Horas de Pesca"} por Hexágono'
            )
            
            # Generar contenido HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Reporte de Análisis VMS - {analysis_type.capitalize()}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
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
                        margin-top: 20px;
                        text-align: center;
                    }}
                    .map-container img {{
                        max-width: 100%;
                        height: auto;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    h1, h2 {{
                        color: #333;
                        border-bottom: 2px solid #eee;
                        padding-bottom: 10px;
                    }}
                </style>
            </head>
            <body>
                <h1>Reporte de Análisis VMS - {analysis_type.capitalize()}</h1>
                
                <h2>Estadísticas Generales</h2>
                <div class="stats-container">
            """
            
            # Agregar estadísticas según el tipo de análisis
            if analysis_type == 'density':
                html_content += f"""
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
                """
            else:  # hours
                html_content += f"""
                    <div class="stat-box">
                        <h3>Horas Totales de Pesca</h3>
                        <p>{stats['total_fishing_hours']:.1f}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Promedio de Horas por Hexágono</h3>
                        <p>{stats['avg_hours_per_hexagon']:.1f}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Máximo de Horas en un Hexágono</h3>
                        <p>{stats['max_hours_in_hexagon']:.1f}</p>
                    </div>
                """
            
            # Agregar estadísticas comunes
            html_content += f"""
                    <div class="stat-box">
                        <h3>Hexágonos Únicos</h3>
                        <p>{stats['unique_hexagons']:,}</p>
                    </div>
                </div>
                
                <h2>Visualización de Actividad</h2>
                <div class="map-container">
                    <img src="temp/activity_map.png" alt="Mapa de Actividad">
                </div>
            </body>
            </html>
            """
            
            # Guardar reporte
            output_file.write_text(html_content, encoding='utf-8')
            logger.info(f"Report generated: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise