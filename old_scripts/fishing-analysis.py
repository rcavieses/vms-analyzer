import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from shapely.geometry import Polygon, Point
from dataclasses import dataclass
from loguru import logger

class FishingAnalyzer:
    def __init__(self, resolution: int = 7):
        """
        Inicializa el analizador de datos pesqueros.
        
        Args:
            resolution: Resolución de los hexágonos H3 (default: 7)
        """
        self.resolution = resolution
        logger.add("fishing_analysis.log", rotation="100 MB")

    def load_data(self, 
                 fishing_data_path: Path,
                 value_data_path: Path,
                 polygon_path: Path,
                 years: List[int]) -> Tuple[pd.DataFrame, float, List[Tuple[float, float]]]:
        """
        Carga todos los datos necesarios para el análisis.
        """
        try:
            # Cargar datos de pesca
            logger.info(f"Cargando datos de pesca desde {fishing_data_path}")
            df = pd.read_csv(fishing_data_path, low_memory=False)
            
            # Cargar datos de valor
            logger.info(f"Cargando datos de valor desde {value_data_path}")
            valor_df = pd.read_csv(value_data_path)
            valor_total = valor_df[valor_df['Year'].isin(years)]['VALOR'].sum()
            logger.info(f"Valor total calculado: ${valor_total:,.2f}")
            
            # Cargar polígono
            logger.info(f"Cargando polígono desde {polygon_path}")
            polygon_df = pd.read_csv(polygon_path)
            polygon_points = list(zip(polygon_df['Longitud'], polygon_df['Latitud']))
            if polygon_points[0] != polygon_points[-1]:
                polygon_points.append(polygon_points[0])
            
            return df, valor_total, polygon_points
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise

    def calculate_fishing_time(self, 
                             df: pd.DataFrame, 
                             polygon_points: List[Tuple[float, float]]) -> gpd.GeoDataFrame:
        """
        Calcula el tiempo de pesca por hexágono.
        """
        try:
            logger.info("Iniciando cálculo de tiempo de pesca")
            
            # Preparar datos
            df = df.copy()
            df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
            
            # Crear GeoDataFrame
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df['Longitud'], df['Latitud']),
                crs='EPSG:4326'
            )
            
            # Asignar índices H3
            gdf['h3_index'] = gdf.apply(
                lambda row: h3.latlng_to_cell(
                    row.geometry.y,
                    row.geometry.x,
                    self.resolution
                ),
                axis=1
            )
            
            # Calcular períodos de pesca
            gdf = gdf.sort_values(['Nombre', 'h3_index', 'Fecha'])
            gdf['time_diff'] = gdf.groupby(['Nombre', 'h3_index'])['Fecha'].diff().dt.total_seconds() / 3600.0
            
            # Identificar períodos continuos
            MAX_GAP = 2  # horas
            gdf['new_period'] = (gdf['time_diff'] > MAX_GAP) | (gdf['time_diff'].isna())
            gdf['period_id'] = gdf.groupby(['Nombre', 'h3_index'])['new_period'].cumsum()
            
            # Calcular tiempo por período
            period_times = gdf.groupby(['h3_index', 'Nombre', 'period_id']).agg({
                'Fecha': ['min', 'max']
            }).reset_index()
            
            period_times['fishing_hours'] = (
                (period_times[('Fecha', 'max')] - period_times[('Fecha', 'min')])
                .dt.total_seconds() / 3600.0
            )
            
            # Agregar por hexágono
            hex_times = period_times.groupby('h3_index')['fishing_hours'].sum().reset_index()
            
            # Generar cobertura completa del polígono
            polygon = Polygon(polygon_points)
            bounds = polygon.bounds
            
            lat_points = np.arange(bounds[1], bounds[3], 0.01)
            lon_points = np.arange(bounds[0], bounds[2], 0.01)
            
            all_hexagons = set()
            for lat in lat_points:
                for lon in lon_points:
                    point = Point(lon, lat)
                    if polygon.contains(point):
                        hex_id = h3.latlng_to_cell(lat, lon, self.resolution)
                        all_hexagons.add(hex_id)
            
            # Asegurar cobertura completa
            all_hex_df = pd.DataFrame({'h3_index': list(all_hexagons)})
            hex_times = pd.merge(all_hex_df, hex_times, on='h3_index', how='left')
            hex_times['fishing_hours'] = hex_times['fishing_hours'].fillna(0)
            
            # Crear GeoDataFrame final
            hex_gdf = gpd.GeoDataFrame(
                hex_times,
                geometry=[Polygon([(vertex[1], vertex[0]) for vertex in h3.cell_to_boundary(h)]) 
                         for h in hex_times['h3_index']],
                crs='EPSG:4326'
            )
            
            logger.info(f"Cálculo completado. Total de hexágonos: {len(hex_gdf)}")
            return hex_gdf
            
        except Exception as e:
            logger.error(f"Error en cálculo de tiempo de pesca: {e}")
            raise

    def calculate_values(self, 
                        hex_gdf: gpd.GeoDataFrame, 
                        valor_total: float) -> gpd.GeoDataFrame:
        """
        Calcula los valores económicos por hexágono.
        """
        try:
            logger.info("Calculando valores económicos")
            
            total_horas = hex_gdf['fishing_hours'].sum()
            valor_por_hora = valor_total / total_horas
            
            hex_gdf['val_millon'] = (hex_gdf['fishing_hours'] * valor_por_hora) / 1_000_000
            hex_gdf['val_millon'] = hex_gdf['val_millon'].round(4)
            
            logger.info(f"Valor promedio por hexágono: ${hex_gdf['val_millon'].mean():,.4f}M")
            return hex_gdf
            
        except Exception as e:
            logger.error(f"Error en cálculo de valores: {e}")
            raise

    def create_map(self, 
                  hex_gdf: gpd.GeoDataFrame, 
                  output_path: Path,
                  title: str = "Mapa de valor por hexágono") -> None:
        """
        Crea y guarda la visualización del mapa.
        """
        try:
            logger.info("Creando visualización del mapa")
            
            # Crear visualización normal y logarítmica
            for scale_type in ['linear', 'log']:
                fig, ax = plt.subplots(figsize=(15, 12))
                
                data = hex_gdf.copy()
                if scale_type == 'log':
                    # Agregar pequeño valor para evitar log(0)
                    min_non_zero = data[data['val_millon'] > 0]['val_millon'].min()
                    data['plot_value'] = data['val_millon'].apply(
                        lambda x: np.log10(x + min_non_zero/10) if x > 0 else 0
                    )
                else:
                    data['plot_value'] = data['val_millon']
                
                # Configurar visualización
                cmap = plt.cm.YlOrRd
                cmap.set_under('lightgrey')
                
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
                        'label': f'Valor (Millones $) - Escala {scale_type}',
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
                plt.title(f'{title} (Escala {scale_type})', pad=20, size=14)
                plt.xlabel('Longitud')
                plt.ylabel('Latitud')
                
                # Guardar
                output_name = output_path.with_stem(f"{output_path.stem}_{scale_type}")
                plt.savefig(output_name, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Mapa guardado: {output_name}")
                
        except Exception as e:
            logger.error(f"Error creando mapa: {e}")
            if plt.get_fignums():
                plt.close()
            raise

def main():
    # Configurar rutas
    base_path = Path(".")
    output_path = base_path / "outputs_nav"
    output_path.mkdir(exist_ok=True)
    
    # Parámetros
    years = [2020, 2021, 2022, 2023]
    resolution = 7
    
    # Rutas de archivos
    fishing_data_path = base_path / "datos_no_pesca.csv"
    value_data_path = base_path / "valor.csv"
    polygon_path = base_path / "verticesGdM.csv"
    
    try:
        # Inicializar analizador
        analyzer = FishingAnalyzer(resolution=resolution)
        
        # Cargar datos
        df, valor_total, polygon_points = analyzer.load_data(
            fishing_data_path,
            value_data_path,
            polygon_path,
            years
        )
        
        # Calcular tiempo de pesca
        hex_gdf = analyzer.calculate_fishing_time(df, polygon_points)
        
        # Calcular valores
        hex_gdf = analyzer.calculate_values(hex_gdf, valor_total)
        
        # Guardar resultados
        # Shapefile
        output_shapefile = output_path / "hexagonos_pesca.shp"
        gdf_to_save = hex_gdf.copy()
        gdf_to_save = gdf_to_save.rename(columns={
            'fishing_hours': 'horas',
            'val_millon': 'val_mill',
            'h3_index': 'h3_idx'
        })
        gdf_to_save.to_file(output_shapefile)
        
        # CSV con datos completos
        output_csv = output_path / "hexagonos_pesca.csv"
        gdf_to_save.drop(columns=['geometry']).to_csv(output_csv, index=False)
        
        # Crear mapas
        output_map = output_path / "mapa_pesca.png"
        analyzer.create_map(hex_gdf, output_map)
        
        logger.info("Análisis completado exitosamente")
        logger.info(f"\nResumen final:")
        logger.info(f"Total de hexágonos: {len(hex_gdf)}")
        logger.info(f"Hexágonos con actividad: {(hex_gdf['val_millon'] > 0).sum()}")
        logger.info(f"Valor total: ${valor_total:,.2f}")
        logger.info(f"Total de horas de pesca: {hex_gdf['fishing_hours'].sum():,.2f}")
        
    except Exception as e:
        logger.error(f"Error en el análisis: {e}")
        raise

if __name__ == "__main__":
    main()
