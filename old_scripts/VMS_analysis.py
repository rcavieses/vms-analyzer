from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import contextily as ctx
from typing import List, Tuple, Optional, Dict, Union
from loguru import logger
import h3
from datetime import datetime

class VMSAnalyzer:
    """
    Biblioteca unificada para el análisis de datos VMS (Vessel Monitoring System).
    Maneja el flujo completo desde la unificación de archivos hasta la visualización
    y análisis espacial.
    """
    
    def __init__(self, 
                 base_path: Path,
                 chunk_size: int = 5_000_000,
                 n_cores: Optional[int] = None):
        """
        Inicializa el analizador VMS.
        
        Args:
            base_path: Directorio base para entrada/salida
            chunk_size: Tamaño de chunks para procesamiento
            n_cores: Número de cores para procesamiento paralelo
        """
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.n_cores = n_cores if n_cores is not None else mp.cpu_count() - 1
        
        # Configuración de directorios
        self.setup_directories()
        
        # Configuración de logging
        self.setup_logging()
        
        # Mapeo estándar de columnas para unificación
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
        
        # Columnas requeridas
        self.required_columns = {
            'Nombre', 'Razón Social', 'Fecha', 'Longitud', 
            'Latitud', 'Velocidad'
        }

    def setup_directories(self) -> None:
        """Configura la estructura de directorios necesaria."""
        self.dirs = {
            'input': self.base_path / 'input',
            'unified': self.base_path / 'unified',
            'filtered': self.base_path / 'filtered',
            'classified': self.base_path / 'classified',
            'analysis': self.base_path / 'analysis',
            'logs': self.base_path / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configura el sistema de logging."""
        log_path = self.dirs['logs'] / f"vms_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.remove()  # Eliminar handler por defecto
        logger.add(log_path, rotation="100 MB")
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

    def unify_files(self, input_folder: Optional[Path] = None) -> Path:
        """
        Unifica múltiples archivos VMS en uno solo, homogenizando las columnas.
        
        Args:
            input_folder: Carpeta con archivos VMS (opcional)
            
        Returns:
            Path al archivo unificado
        """
        input_folder = input_folder or self.dirs['input']
        output_file = self.dirs['unified'] / 'vms_unified.csv'
        
        try:
            # Obtener lista de archivos
            files = []
            for ext in ['.csv', '.xlsx']:
                files.extend(input_folder.glob(f'*{ext}'))
            
            if not files:
                raise ValueError(f"No se encontraron archivos en {input_folder}")
            
            logger.info(f"Procesando {len(files)} archivos...")
            
            # Procesar primer archivo
            first_df = self._process_single_file(files[0])
            first_df.to_csv(output_file, index=False, encoding='utf-8')
            total_rows = len(first_df)
            
            # Procesar resto de archivos
            for file_path in tqdm(files[1:], desc="Unificando archivos"):
                df = self._process_single_file(file_path)
                if df is not None:
                    df.to_csv(output_file, mode='a', header=False, 
                            index=False, encoding='utf-8')
                    total_rows += len(df)
            
            logger.info(f"Unificación completada: {total_rows:,} registros totales")
            return output_file
            
        except Exception as e:
            logger.error(f"Error en unificación: {e}")
            raise

    def _process_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Procesa un archivo individual para la unificación."""
        try:
            # Leer archivo
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            else:
                df = pd.read_excel(file_path)
            
            # Homogenizar columnas
            df.columns = df.columns.str.strip()
            df.rename(columns=self.column_mapping, inplace=True)
            
            # Validar y limpiar datos
            df = df.dropna(subset=['Longitud', 'Latitud', 'Fecha'])
            
            # Verificar columnas requeridas
            missing_cols = self.required_columns - set(df.columns)
            if missing_cols:
                logger.warning(f"Columnas faltantes en {file_path.name}: {missing_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error procesando {file_path.name}: {e}")
            return None

    def filter_by_polygon(self, 
                         input_file: Optional[Path] = None,
                         vertices_file: Optional[Path] = None) -> Path:
        """
        Filtra los datos VMS según un polígono definido.
        
        Args:
            input_file: Archivo VMS a filtrar (opcional)
            vertices_file: Archivo con vértices del polígono (opcional)
            
        Returns:
            Path al archivo filtrado
        """
        input_file = input_file or self.dirs['unified'] / 'vms_unified.csv'
        vertices_file = vertices_file or self.dirs['input'] / 'vertices.csv'
        output_file = self.dirs['filtered'] / 'vms_filtered.csv'
        
        try:
            # Cargar polígono
            vertices_df = pd.read_csv(vertices_file)
            vertices = [(row['Longitud'], row['Latitud']) 
                       for _, row in vertices_df.iterrows()]
            if vertices[0] != vertices[-1]:
                vertices.append(vertices[0])
            polygon = Polygon(vertices)
            
            # Configurar procesamiento paralelo
            pool = mp.Pool(self.n_cores)
            
            # Procesar en chunks
            filtered_chunks = []
            with tqdm(desc="Filtrando por polígono") as pbar:
                reader = pd.read_csv(input_file, chunksize=self.chunk_size)
                chunk_pairs = ((chunk, polygon) for chunk in reader)
                
                for result in pool.imap(self._process_polygon_chunk, chunk_pairs):
                    if result is not None:
                        filtered_chunks.append(result)
                    pbar.update(1)
            
            pool.close()
            pool.join()
            
            # Unir resultados
            if filtered_chunks:
                final_result = pd.concat(filtered_chunks, ignore_index=True)
                final_result.to_csv(output_file, index=False)
                logger.info(f"Filtrado completado: {len(final_result):,} registros")
                return output_file
            else:
                raise ValueError("No se encontraron datos dentro del polígono")
                
        except Exception as e:
            logger.error(f"Error en filtrado: {e}")
            raise

    @staticmethod
    def _process_polygon_chunk(args: Tuple[pd.DataFrame, Polygon]) -> Optional[gpd.GeoDataFrame]:
        """Procesa un chunk para el filtrado por polígono."""
        chunk, polygon = args
        try:
            # Crear GeoDataFrame
            geometry = gpd.points_from_xy(chunk['Longitud'], chunk['Latitud'])
            points_gdf = gpd.GeoDataFrame(chunk, geometry=geometry, crs="EPSG:4326")
            
            # Filtrar por polígono
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
            result = gpd.sjoin(points_gdf, polygon_gdf, predicate='within')
            
            return result.drop(columns=['index_right']) if 'index_right' in result.columns else result
            
        except Exception:
            return None

    def classify_fishing_activity(self,
                                input_file: Optional[Path] = None,
                                speed_range: Tuple[float, float] = (0, 6)) -> Path:
        """
        Clasifica la actividad pesquera según la velocidad.
        
        Args:
            input_file: Archivo a clasificar (opcional)
            speed_range: Rango de velocidades para pesca (min, max)
            
        Returns:
            Path al archivo clasificado
        """
        input_file = input_file or self.dirs['filtered'] / 'vms_filtered.csv'
        output_file = self.dirs['classified'] / 'vms_classified.csv'
        
        try:
            # Cargar y procesar datos
            df = pd.read_csv(input_file)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            
            # Clasificar actividad
            df['Actividad'] = df['Velocidad'].apply(
                lambda x: 'Pesca' if speed_range[0] <= x <= speed_range[1] else 'No Pesca'
            )
            
            # Guardar resultados
            df.to_csv(output_file, index=False)
            
            # Generar estadísticas
            stats = {
                'Total registros': len(df),
                'Registros de pesca': (df['Actividad'] == 'Pesca').sum(),
                'Registros de no pesca': (df['Actividad'] == 'No Pesca').sum(),
                'Embarcaciones únicas': df['Nombre'].nunique()
            }
            
            logger.info("\nEstadísticas de clasificación:")
            for key, value in stats.items():
                logger.info(f"{key}: {value:,}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            raise

    def analyze_fishing_effort(self,
                             input_file: Optional[Path] = None,
                             h3_resolution: int = 7,
                             valor_file: Optional[Path] = None) -> Dict[str, Path]:
        """
        Analiza el esfuerzo pesquero usando índices H3 y genera visualizaciones.
        
        Args:
            input_file: Archivo clasificado a analizar (opcional)
            h3_resolution: Resolución de hexágonos H3
            valor_file: Archivo con datos de valor económico (opcional)
            
        Returns:
            Diccionario con paths a los archivos generados
        """
        input_file = input_file or self.dirs['classified'] / 'vms_classified.csv'
        output_prefix = self.dirs['analysis'] / 'fishing_effort'
        
        try:
            # Cargar datos
            df = pd.read_csv(input_file)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            
            # Calcular índices H3
            df['h3_index'] = df.apply(
                lambda row: h3.geo_to_h3(row['Latitud'], row['Longitud'], h3_resolution),
                axis=1
            )
            
            # Calcular tiempo por hexágono
            hex_times = self._calculate_hex_times(df)
            
            # Crear GeoDataFrame
            hex_gdf = self._create_hex_gdf(hex_times)
            
            # Calcular valores económicos si se proporciona archivo
            if valor_file:
                hex_gdf = self._add_economic_values(hex_gdf, valor_file)
            
            # Guardar resultados
            outputs = {}
            
            # Shapefile
            shp_path = output_prefix.with_suffix('.shp')
            hex_gdf.to_file(shp_path)
            outputs['shapefile'] = shp_path
            
            # CSV
            csv_path = output_prefix.with_suffix('.csv')
            hex_gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
            outputs['csv'] = csv_path
            
            # Mapas
            for scale in ['linear', 'log']:
                map_path = output_prefix.with_name(f"{output_prefix.stem}_{scale}.png")
                self._create_effort_map(hex_gdf, map_path, scale=scale)
                outputs[f'map_{scale}'] = map_path
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error en análisis: {e}")
            raise

    def _calculate_hex_times(self, df: pd.DataFrame) -> pd.DataFrame:
            """Calcula tiempo de pesca por hexágono."""
            df = df.sort_values(['Nombre', 'h3_index', 'Fecha'])
            df['time_diff'] = df.groupby(['Nombre', 'h3_index'])['Fecha'].diff().dt.total_seconds() / 3600.0
            
            # Identificar períodos continuos
            MAX_GAP = 2  # horas
            df['new_period'] = (df['time_diff'] > MAX_GAP) | (df['time_diff'].isna())
            df['period_id'] = df.groupby(['Nombre', 'h3_index'])['new_period'].cumsum()
            
            # Calcular tiempo por período
            period_times = df.groupby(['h3_index', 'Nombre', 'period_id']).agg({
                'Fecha': ['min', 'max']
            }).reset_index()
            
            # Calcular duración de cada período
            period_times['fishing_hours'] = (
                (period_times[('Fecha', 'max')] - period_times[('Fecha', 'min')])
                .dt.total_seconds() / 3600.0
            )
            
            # Agregar por hexágono
            hex_times = period_times.groupby('h3_index')['fishing_hours'].sum().reset_index()
            
            return hex_times

    def _create_hex_gdf(self, hex_times: pd.DataFrame) -> gpd.GeoDataFrame:
            """
            Crea un GeoDataFrame con geometrías de hexágonos H3.
            
            Args:
                hex_times: DataFrame con tiempos por hexágono
                
            Returns:
                GeoDataFrame con geometrías de hexágonos
            """
            # Convertir índices H3 a polígonos
            geometries = []
            for h3_idx in hex_times['h3_index']:
                # Obtener vértices del hexágono
                vertices = h3.h3_to_geo_boundary(h3_idx, geo_json=True)
                # Crear polígono (invertir coordenadas ya que h3 devuelve [lat, lng])
                geometries.append(Polygon([(v[1], v[0]) for v in vertices]))
            
            # Crear GeoDataFrame
            hex_gdf = gpd.GeoDataFrame(
                hex_times,
                geometry=geometries,
                crs="EPSG:4326"
            )
            
            return hex_gdf

    def _add_economic_values(self, 
                            hex_gdf: gpd.GeoDataFrame,
                            valor_file: Path) -> gpd.GeoDataFrame:
            """
            Añade valores económicos al GeoDataFrame de hexágonos.
            
            Args:
                hex_gdf: GeoDataFrame con tiempos de pesca
                valor_file: Archivo con datos de valor económico
                
            Returns:
                GeoDataFrame con valores económicos añadidos
            """
            try:
                # Cargar datos de valor
                valor_df = pd.read_csv(valor_file)
                valor_total = valor_df['VALOR'].sum()
                
                # Calcular valor por hora
                total_horas = hex_gdf['fishing_hours'].sum()
                if total_horas <= 0:
                    raise ValueError("Total de horas es 0 o negativo")
                    
                valor_por_hora = valor_total / total_horas
                
                # Calcular valor por hexágono
                hex_gdf['val_millon'] = (hex_gdf['fishing_hours'] * valor_por_hora) / 1_000_000
                hex_gdf['val_millon'] = hex_gdf['val_millon'].round(4)
                
                return hex_gdf
                
            except Exception as e:
                logger.error(f"Error calculando valores económicos: {e}")
                raise

    def _create_effort_map(self,
                            hex_gdf: gpd.GeoDataFrame,
                            output_path: Path,
                            scale: str = 'linear',
                            title: Optional[str] = None) -> None:
            """
            Crea y guarda un mapa de esfuerzo pesquero.
            
            Args:
                hex_gdf: GeoDataFrame con datos de esfuerzo
                output_path: Ruta para guardar el mapa
                scale: Tipo de escala ('linear' o 'log')
                title: Título opcional para el mapa
            """
            try:
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Preparar datos para visualización
                data = hex_gdf.copy()
                column = 'val_millon' if 'val_millon' in data.columns else 'fishing_hours'
                
                if scale == 'log':
                    # Agregar pequeño valor para evitar log(0)
                    min_non_zero = data[data[column] > 0][column].min()
                    data['plot_value'] = data[column].apply(
                        lambda x: np.log10(x + min_non_zero/10) if x > 0 else 0
                    )
                else:
                    data['plot_value'] = data[column]
                
                # Configurar visualización
                cmap = plt.cm.YlOrRd
                cmap.set_under('lightgrey')
                
                # Calcular límites
                non_zero_values = data[data['plot_value'] > 0]['plot_value']
                if len(non_zero_values) > 0:
                    vmin = non_zero_values.min()
                    vmax = non_zero_values.quantile(0.99)
                else:
                    vmin = 0
                    vmax = 1
                
                # Crear plot
                data.plot(
                    column='plot_value',
                    cmap=cmap,
                    legend=True,
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    legend_kwds={
                        'label': f'{"Valor (Millones $)" if column == "val_millon" else "Horas de pesca"} - Escala {scale}',
                        'orientation': 'vertical',
                        'extend': 'both'
                    }
                )
                
                # Agregar mapa base
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.Esri.WorldImagery,
                    zoom=8
                )
                
                # Personalizar plot
                if title is None:
                    title = f'Mapa de {"valor" if column == "val_millon" else "esfuerzo pesquero"} (Escala {scale})'
                plt.title(title, pad=20, size=14)
                plt.xlabel('Longitud')
                plt.ylabel('Latitud')
                
                # Guardar
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Mapa guardado en {output_path}")
                
            except Exception as e:
                logger.error(f"Error creando mapa: {e}")
                if plt.get_fignums():
                    plt.close()
                raise

