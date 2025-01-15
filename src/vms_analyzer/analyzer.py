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
                        speed_threshold: float = 2.0) -> Tuple[gpd.GeoDataFrame, Dict]:
        """
        Analyzes vessel activity patterns using H3 hexagons.
        
        Args:
            input_file: VMS file to analyze (optional)
            resolution: H3 hexagon resolution (default: 8)
            speed_threshold: Speed threshold for activity classification (default: 2.0)
            
        Returns:
            Tuple containing:
            - GeoDataFrame with hexagon geometries and metrics
            - Dictionary with summary statistics
        """
        input_file = input_file or self.dirs['filtered'] / 'vms_filtered.csv'
        
        try:
            logger.info("Starting activity analysis...")
            
            # Process data in chunks
            hex_counts = {}
            total_points = 0
            active_points = 0
            
            for chunk in tqdm(pd.read_csv(input_file, chunksize=self.chunk_size),
                            desc="Processing chunks"):
                # Convert coordinates to H3 indexes
                h3_indexes = [
                    h3.geo_to_h3(lat, lon, resolution)
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
            
            # Create hexagon geometries
            hexagons = []
            for h3_idx, counts in hex_counts.items():
                # Get hexagon boundaries
                boundaries = h3.h3_to_geo_boundary(h3_idx)
                poly = Polygon(boundaries)
                
                # Calculate activity percentage
                activity_pct = (counts['active'] / counts['total']) * 100
                
                hexagons.append({
                    'geometry': poly,
                    'h3_index': h3_idx,
                    'total_points': counts['total'],
                    'active_points': counts['active'],
                    'activity_percentage': activity_pct
                })
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(hexagons)
            
            # Calculate summary statistics
            stats = {
                'total_points': total_points,
                'active_points': active_points,
                'activity_percentage': (active_points / total_points) * 100,
                'unique_hexagons': len(hex_counts),
                'avg_points_per_hexagon': total_points / len(hex_counts)
            }
            
            logger.info("Activity analysis completed")
            return gdf, stats
            
        except Exception as e:
            logger.error(f"Error in activity analysis: {e}")
            raise

    def generate_report(self, 
                       gdf: gpd.GeoDataFrame, 
                       stats: Dict,
                       output_file: Optional[Path] = None) -> Path:
        """
        Generates an HTML report with analysis results.
        
        Args:
            gdf: GeoDataFrame with analysis results
            stats: Dictionary with summary statistics
            output_file: Path for output HTML file (optional)
            
        Returns:
            Path to generated report
        """
        output_file = output_file or self.dirs['reports'] / f'vms_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        try:
            # Create map visualization
            m = utils.create_map(gdf)
            
            # Generate HTML report
            html_content = utils.generate_html_report(m, stats)
            
            # Save report
            output_file.write_text(html_content, encoding='utf-8')
            logger.info(f"Report generated: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise