import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Optional
import sys
import os
from datetime import datetime

class VMSOptimizer:
    def __init__(self, 
                 input_file: Path,
                 vertices_file: Path,
                 output_folder: Path,
                 chunk_size: int = 5_000_000,
                 n_cores: Optional[int] = None):
        """
        Inicializa el optimizador de datos VMS.
        
        Args:
            input_file: Ruta al archivo VMS de entrada
            vertices_file: Ruta al archivo de vértices del polígono
            output_folder: Ruta para guardar resultados
            chunk_size: Tamaño de los chunks para procesamiento
            n_cores: Número de cores a utilizar (None para automático)
        """
        self.input_file = input_file
        self.vertices_file = vertices_file
        self.output_folder = output_folder
        self.chunk_size = chunk_size
        self.n_cores = n_cores if n_cores is not None else mp.cpu_count() - 1
        
        # Crear directorio de salida
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        log_path = self.output_folder / f"vms_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.remove()  # Remover handler por defecto
        logger.add(log_path, rotation="100 MB")
        logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

    def load_polygon(self) -> Polygon:
        """Carga y valida el polígono desde el archivo de vértices."""
        try:
            logger.info(f"Cargando vértices desde {self.vertices_file}")
            vertices_df = pd.read_csv(self.vertices_file, low_memory=False)
            
            required_columns = {'Longitud', 'Latitud'}
            if not required_columns.issubset(vertices_df.columns):
                raise ValueError(f"Columnas faltantes en archivo de vértices: {required_columns - set(vertices_df.columns)}")
            
            vertices = [(row['Longitud'], row['Latitud']) for _, row in vertices_df.iterrows()]
            
            # Asegurar que el polígono esté cerrado
            if vertices[0] != vertices[-1]:
                vertices.append(vertices[0])
            
            polygon = Polygon(vertices)
            if not polygon.is_valid:
                raise ValueError("El polígono construido no es válido")
            
            logger.info(f"Polígono cargado exitosamente con {len(vertices)} vértices")
            return polygon
            
        except Exception as e:
            logger.error(f"Error cargando polígono: {e}")
            raise

    @staticmethod
    def create_geometry_chunk(df_chunk: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convierte un chunk de datos a GeoDataFrame."""
        try:
            geometry = gpd.points_from_xy(df_chunk['Longitud'], df_chunk['Latitud'])
            return gpd.GeoDataFrame(df_chunk, geometry=geometry, crs="EPSG:4326")
        except Exception as e:
            logger.error(f"Error creando geometría: {e}")
            raise

    @staticmethod
    def process_chunk(args: Tuple[pd.DataFrame, Polygon]) -> Optional[gpd.GeoDataFrame]:
        """Procesa un chunk individual de datos."""
        chunk_data, polygon = args
        try:
            # Validación básica de datos
            if chunk_data.empty:
                return None
                
            # Filtrar valores inválidos
            chunk_data = chunk_data.dropna(subset=['Longitud', 'Latitud'])
            
            # Convertir a GeoDataFrame
            points_gdf = VMSOptimizer.create_geometry_chunk(chunk_data)
            
            # Crear GeoDataFrame del polígono
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
            
            # Join espacial
            result = gpd.sjoin(points_gdf, polygon_gdf, predicate='within')
            
            # Limpiar columnas
            if 'index_right' in result.columns:
                result = result.drop(columns=['index_right'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando chunk: {e}")
            return None

    def optimize(self) -> None:
        """Ejecuta el proceso completo de optimización."""
        try:
            # 1. Cargar polígono
            polygon = self.load_polygon()
            
            # 2. Configurar pool de procesamiento
            logger.info(f"Iniciando procesamiento paralelo con {self.n_cores} cores")
            pool = mp.Pool(self.n_cores)
            
            # 3. Contar chunks totales
            logger.info("Calculando número total de chunks...")
            total_chunks = sum(1 for _ in pd.read_csv(self.input_file, chunksize=self.chunk_size, low_memory=False))
            logger.info(f"Total de chunks a procesar: {total_chunks}")
            
            # 4. Procesar chunks
            filtered_chunks = []
            total_rows_processed = 0
            total_rows_kept = 0
            
            logger.info("Iniciando procesamiento de chunks...")
            with tqdm(total=total_chunks, desc="Procesando chunks") as pbar:
                reader = pd.read_csv(self.input_file, chunksize=self.chunk_size, low_memory=False)
                chunk_polygon_pairs = ((chunk, polygon) for chunk in reader)
                
                for chunk_result in pool.imap(self.process_chunk, chunk_polygon_pairs):
                    if chunk_result is not None:
                        total_rows_processed += self.chunk_size
                        total_rows_kept += len(chunk_result)
                        filtered_chunks.append(chunk_result)
                    pbar.update(1)
            
            # 5. Cerrar pool
            pool.close()
            pool.join()
            
            # 6. Concatenar y guardar resultados
            if filtered_chunks:
                logger.info("Concatenando resultados...")
                final_result = pd.concat(filtered_chunks, ignore_index=True)
                
                # Optimizar tipos de datos
                logger.info("Optimizando tipos de datos...")
                for col in final_result.select_dtypes(include=['float64']).columns:
                    final_result[col] = pd.to_numeric(final_result[col], downcast='float')
                
                # Guardar resultados
                output_file = self.output_folder / "vms_GdM_optimized.csv"
                logger.info(f"Guardando resultados en {output_file}")
                final_result.to_csv(output_file, index=False)
                
                # Generar reporte
                logger.info("\nReporte final:")
                logger.info(f"Total de registros procesados: {total_rows_processed:,}")
                logger.info(f"Registros dentro del polígono: {total_rows_kept:,}")
                logger.info(f"Porcentaje de registros conservados: {(total_rows_kept/total_rows_processed)*100:.2f}%")
                logger.info(f"Tamaño del archivo resultante: {os.path.getsize(output_file)/(1024*1024):.2f} MB")
                
            else:
                logger.warning("No se encontraron datos dentro del polígono")
            
        except Exception as e:
            logger.error(f"Error en el proceso de optimización: {e}")
            raise
        
        finally:
            logger.info("Proceso completado")

def main():
    # Configurar rutas
    base_path = Path(".")
    input_file = base_path / "vms_unidos.csv"
    vertices_file = base_path / "vertices.csv"
    output_folder = base_path / "optimized_data"
    
    # Parámetros
    chunk_size = 5_000_000  # 5 millones de registros por chunk
    n_cores = None  # Usar configuración automática
    
    try:
        # Inicializar y ejecutar optimizador
        optimizer = VMSOptimizer(
            input_file=input_file,
            vertices_file=vertices_file,
            output_folder=output_folder,
            chunk_size=chunk_size,
            n_cores=n_cores
        )
        
        optimizer.optimize()
        
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
