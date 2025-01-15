import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from loguru import logger
from typing import Optional, Tuple

class VMSProcessor:
    def __init__(self, input_path: Path, output_path: Path):
        """
        Inicializa el procesador de datos VMS.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_path: Ruta para guardar los resultados
        """
        self.input_path = input_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        log_path = output_path / "vms_processing.log"
        logger.add(log_path, rotation="100 MB")

    def load_and_clean_data(self, 
                           velocidad_pesca: Tuple[float, float] = (0, 6),
                           required_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Carga y limpia los datos VMS.
        
        Args:
            velocidad_pesca: Rango de velocidades consideradas como pesca (min, max)
            required_columns: Lista de columnas requeridas
        """
        try:
            logger.info(f"Cargando datos desde {self.input_path}")
            df = pd.read_csv(self.input_path, low_memory=False)
            
            # Verificar columnas requeridas
            if required_columns is None:
                required_columns = ['Longitud', 'Latitud', 'Velocidad', 'Fecha', 'Nombre']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columnas faltantes: {missing_columns}")
            
            # Limpiar datos
            logger.info("Limpiando datos...")
            
            # Eliminar duplicados
            df = df.drop_duplicates()
            
            # Verificar y limpiar coordenadas
            df = df[
                (df['Longitud'].between(-98, -82)) &
                (df['Latitud'].between(18, 31))
            ]
            
            # Verificar y limpiar velocidades
            df = df[df['Velocidad'].notna()]
            
            # Clasificar actividad pesquera
            df['Clasificacion'] = df['Velocidad'].apply(
                lambda x: 'Pesca' if velocidad_pesca[0] <= x <= velocidad_pesca[1] else 'No Pesca'
            )
            
            logger.info(f"Datos procesados: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Error procesando datos: {e}")
            raise

    def create_verification_map(self, df: pd.DataFrame, save_path: Path) -> None:
        """
        Crea un mapa de verificación de los datos procesados.
        """
        try:
            logger.info("Creando mapa de verificación...")
            
            fig, ax = plt.subplots(
                figsize=(15, 10),
                subplot_kw={'projection': ccrs.Mercator()}
            )
            
            # Configurar mapa base
            ax.set_extent([-98, -82, 18, 31], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='white')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            
            # Separar datos por clasificación
            pesca = df['Clasificacion'] == 'Pesca'
            no_pesca = df['Clasificacion'] == 'No Pesca'
            
            # Graficar puntos
            ax.scatter(
                df.loc[pesca, 'Longitud'],
                df.loc[pesca, 'Latitud'],
                c='blue', s=1, alpha=0.6,
                label='Pesca',
                transform=ccrs.PlateCarree()
            )
            
            ax.scatter(
                df.loc[no_pesca, 'Longitud'],
                df.loc[no_pesca, 'Latitud'],
                c='red', s=1, alpha=0.6,
                label='No Pesca',
                transform=ccrs.PlateCarree()
            )
            
            # Añadir elementos del mapa
            ax.gridlines(draw_labels=True)
            plt.title('Verificación de Clasificación de Actividad Pesquera')
            plt.legend(loc='lower left')
            
            # Guardar mapa
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Mapa guardado en {save_path}")
            
        except Exception as e:
            logger.error(f"Error creando mapa: {e}")
            if plt.get_fignums():
                plt.close()
            raise

    def process_and_save(self, 
                        velocidad_pesca: Tuple[float, float] = (0, 6),
                        required_columns: Optional[list] = None) -> None:
        """
        Procesa los datos VMS y guarda los resultados.
        """
        try:
            # Cargar y procesar datos
            df = self.load_and_clean_data(velocidad_pesca, required_columns)
            
            # Guardar datos procesados
            output_csv = self.output_path / "datos_pesca.csv"
            df.to_csv(output_csv, index=False)
            logger.info(f"Datos guardados en {output_csv}")
            
            # Crear mapa de verificación
            output_map = self.output_path / "verificacion_pesca.png"
            self.create_verification_map(df, output_map)
            
            # Generar resumen
            summary = {
                'Total registros': len(df),
                'Registros de pesca': (df['Clasificacion'] == 'Pesca').sum(),
                'Registros de no pesca': (df['Clasificacion'] == 'No Pesca').sum(),
                'Rango de fechas': f"{df['Fecha'].min()} - {df['Fecha'].max()}",
                'Número de embarcaciones': df['Nombre'].nunique()
            }
            
            logger.info("\nResumen del procesamiento:")
            for key, value in summary.items():
                logger.info(f"{key}: {value}")
            
        except Exception as e:
            logger.error(f"Error en el procesamiento: {e}")
            raise

def main():
    # Configurar rutas
    base_path = Path(".")
    input_file = base_path / "vms_GdM_optimized.csv"
    output_folder = base_path / "processed_data"
    
    # Parámetros de procesamiento
    velocidad_pesca = (0, 6)  # Rango de velocidades para pesca
    required_columns = ['Longitud', 'Latitud', 'Velocidad', 'Fecha', 'Nombre']
    
    try:
        # Inicializar procesador
        processor = VMSProcessor(input_file, output_folder)
        
        # Procesar datos
        processor.process_and_save(velocidad_pesca, required_columns)
        
        logger.info("Procesamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el script: {e}")
        raise

if __name__ == "__main__":
    main()
