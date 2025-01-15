import pandas as pd
import os
from pathlib import Path
from typing import Optional, List, Iterator
from loguru import logger
from datetime import datetime
import sys

class VMSUnifier:
    """Clase para homogenizar y unir archivos de datos VMS."""
    
    def __init__(self, input_folder: Path, output_folder: Path):
        """
        Inicializa el unificador de datos VMS.
        
        Args:
            input_folder: Carpeta con los archivos a procesar
            output_folder: Carpeta donde se guardarán los resultados
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        log_path = self.output_folder / f"vms_unification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.remove()
        logger.add(log_path, rotation="100 MB")
        logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
        
        # Mapeo de columnas
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
        
        # Columnas a eliminar
        self.columns_to_remove = ['Matricula', 'Tipo_Mensaje']
        
        # Columnas requeridas
        self.required_columns = {'Nombre', 'Razón Social', 'Fecha', 'Longitud', 'Latitud', 'Velocidad'}

    def get_files_to_process(self) -> List[Path]:
        """Obtiene la lista de archivos CSV y Excel en la carpeta de entrada."""
        files = []
        for extension in ['.csv', '.xlsx']:
            files.extend(self.input_folder.glob(f'*{extension}'))
        return sorted(files)

    def read_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Lee un archivo CSV o Excel."""
        try:
            if file_path.suffix == '.csv':
                for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                        logger.info(f"Archivo {file_path.name} leído con codificación {encoding}")
                        return df
                    except UnicodeDecodeError:
                        continue
                logger.error(f"No se pudo leer {file_path.name} con ninguna codificación")
                return None
                
            elif file_path.suffix == '.xlsx':
                return pd.read_excel(file_path)
                
        except Exception as e:
            logger.error(f"Error leyendo {file_path.name}: {e}")
            return None

    def homogenize_columns(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Homogeniza las columnas del DataFrame."""
        try:
            original_columns = set(df.columns)
            
            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip()
            
            # Aplicar mapeo
            df.rename(columns=self.column_mapping, inplace=True)
            
            # Eliminar columnas no deseadas
            df.drop(columns=[col for col in self.columns_to_remove if col in df.columns], inplace=True)
            
            # Verificar columnas requeridas
            missing_columns = self.required_columns - set(df.columns)
            if missing_columns:
                logger.warning(f"Columnas faltantes en {filename}: {missing_columns}")
            
            # Logging
            logger.info(f"Cambios en columnas para {filename}:")
            logger.info(f"  Original: {original_columns}")
            logger.info(f"  Final: {set(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error homogenizando columnas en {filename}: {e}")
            raise

    def process_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Procesa un archivo individual."""
        try:
            logger.info(f"Procesando {file_path.name}")
            
            # Leer archivo
            df = self.read_file(file_path)
            if df is None:
                return None
            
            # Homogenizar columnas
            df = self.homogenize_columns(df, file_path.name)
            
            # Verificar datos
            df = df.dropna(subset=['Longitud', 'Latitud', 'Fecha'])
            
            # Añadir columna con fuente del archivo
            df['archivo_fuente'] = file_path.name
            
            logger.info(f"Registros procesados en {file_path.name}: {len(df):,}")
            return df
            
        except Exception as e:
            logger.error(f"Error procesando {file_path.name}: {e}")
            return None

    def process_and_merge(self, chunk_size: int = 100_000) -> None:
        """Procesa todos los archivos y los une en uno solo."""
        try:
            files = self.get_files_to_process()
            if not files:
                logger.error("No se encontraron archivos para procesar")
                return
                
            logger.info(f"Encontrados {len(files)} archivos para procesar")
            
            # Archivo de salida
            output_file = self.output_folder / "vms_unidos.csv"
            
            # Procesar primer archivo para obtener encabezados
            first_df = self.process_file(files[0])
            if first_df is None:
                raise ValueError("Error procesando el primer archivo")
                
            # Escribir primer archivo con encabezados
            first_df.to_csv(output_file, index=False, encoding='utf-8')
            total_rows = len(first_df)
            
            # Procesar resto de archivos
            for file_path in files[1:]:
                df = self.process_file(file_path)
                if df is None:
                    continue
                
                # Escribir en chunks para manejar memoria
                for chunk_start in range(0, len(df), chunk_size):
                    chunk = df.iloc[chunk_start:chunk_start + chunk_size]
                    chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
                
                total_rows += len(df)
                logger.info(f"Progreso: {total_rows:,} registros totales")
            
            logger.success(f"\nProceso completado:")
            logger.info(f"Archivos procesados: {len(files)}")
            logger.info(f"Total registros unidos: {total_rows:,}")
            logger.info(f"Archivo final: {output_file}")
            
        except Exception as e:
            logger.error(f"Error en el proceso de unificación: {e}")
            raise

def main():
    # Configurar rutas
    base_path = Path(".")
    input_folder = base_path / "archivos_vms"  # Carpeta con los archivos a procesar
    output_folder = base_path / "datos_unidos"  # Carpeta para los resultados
    
    try:
        # Inicializar unificador
        unifier = VMSUnifier(input_folder, output_folder)
        
        # Procesar y unir archivos
        unifier.process_and_merge()
        
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
