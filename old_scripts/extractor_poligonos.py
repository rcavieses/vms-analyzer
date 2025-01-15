import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

# Número de cores a utilizar (dejando uno libre para el sistema)
N_CORES = mp.cpu_count() - 1
CHUNK_SIZE = 5_000_000

def create_geometry_chunk(df_chunk):
    """Convierte un chunk de datos a GeoDataFrame de manera eficiente"""
    geometry = gpd.points_from_xy(df_chunk['Longitud'], df_chunk['Latitud'])
    return gpd.GeoDataFrame(df_chunk, geometry=geometry, crs="EPSG:4326")

def process_chunk(args):
    """Procesa un chunk individual y retorna los puntos dentro del polígono"""
    chunk_data, polygon = args
    try:
        # Convertir chunk a GeoDataFrame
        points_gdf = create_geometry_chunk(chunk_data)
        
        # Crear GeoDataFrame del polígono en este proceso
        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        
        # Realizar join espacial con el polígono
        result = gpd.sjoin(points_gdf, polygon_gdf, predicate='within')
        
        # Limpiar columnas del join que no necesitamos
        if 'index_right' in result.columns:
            result = result.drop(columns=['index_right'])
        
        return result
    except Exception as e:
        print(f"Error procesando chunk: {e}")
        return None

if __name__ == '__main__':
    # 1. Cargar el polígono
    print("Cargando vértices del polígono...")
    vertices_df = pd.read_csv('vertices.csv', low_memory=False)
    vertices = [(row['Longitud'], row['Latitud']) for index, row in vertices_df.iterrows()]
    polygon = Polygon(vertices)
    
    # 2. Configurar el pool de procesamiento
    pool = mp.Pool(N_CORES)
    
    # 3. Leer y procesar en chunks con barra de progreso
    print("Contando número total de chunks...")
    total_rows = sum(1 for _ in pd.read_csv('datos_pesca.csv', chunksize=CHUNK_SIZE, low_memory=False))
    
    filtered_chunks = []
    print("Iniciando procesamiento paralelo...")
    with tqdm(total=total_rows) as pbar:
        # Crear un iterador que combine cada chunk con el polígono
        reader = pd.read_csv('datos_pesca.csv', chunksize=CHUNK_SIZE, low_memory=False)
        chunk_polygon_pairs = ((chunk, polygon) for chunk in reader)
        
        # Procesar chunks en paralelo
        for chunk_result in pool.imap(process_chunk, chunk_polygon_pairs):
            if chunk_result is not None:
                filtered_chunks.append(chunk_result)
            pbar.update(1)
    
    # 4. Cerrar el pool
    pool.close()
    pool.join()
    
    # 5. Concatenar resultados y guardar
    print("Concatenando resultados...")
    if filtered_chunks:
        final_result = pd.concat(filtered_chunks, ignore_index=True)
        
        # 6. Guardar resultados optimizando el espacio
        print("Guardando resultados...")
        final_result.to_csv('vms_AE_pesca.csv', index=False)
        print("¡Proceso completado!")
    else:
        print("No se encontraron resultados para guardar.")