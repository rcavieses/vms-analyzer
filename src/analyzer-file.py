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
            filtered_chunks = []
            with tqdm(desc="