from pathlib import Path
from vms_analyzer import VMSAnalyzer
from loguru import logger

def run_full_analysis():
    """Ejecuta el flujo completo de análisis."""
    base_path = Path("./vms_analysis")
    
    # Inicializar analizador
    analyzer = VMSAnalyzer(base_path)
    
    try:
        # 1. Unificar archivos
        unified_file = analyzer.unify_files()
        
        # 2. Filtrar por polígono
        filtered_file = analyzer.filter_by_polygon(
            input_file=unified_file,
            vertices_file=base_path / "input/vertices.csv"
        )
        
        # 3. Clasificar actividad pesquera
        classified_file = analyzer.classify_fishing_activity(
            input_file=filtered_file,
            speed_range=(0, 6)
        )
        
        # 4. Analizar esfuerzo pesquero
        results = analyzer.analyze_fishing_effort(
            input_file=classified_file,
            h3_resolution=7,
            valor_file=base_path / "input/valor.csv"
        )
        
        logger.info("Análisis completado exitosamente")
        logger.info("\nArchivos generados:")
        for key, path in results.items():
            logger.info(f"{key}: {path}")
            
    except Exception as e:
        logger.error(f"Error en el proceso: {e}")

def run_partial_analysis():
    """Ejemplo de uso parcial de la biblioteca."""
    base_path = Path("./otro_analisis")
    
    # Inicializar con configuración personalizada
    analyzer = VMSAnalyzer(
        base_path=base_path,
        chunk_size=1_000_000,  # Chunks más pequeños
        n_cores=4              # Número específico de cores
    )
    
    try:
        # Solo clasificar actividad de un archivo existente
        classified_file = analyzer.classify_fishing_activity(
            input_file=base_path / "datos_existentes.csv",
            speed_range=(1, 5)  # Rango de velocidades diferente
        )
        
        # Analizar sin datos económicos
        results = analyzer.analyze_fishing_effort(
            input_file=classified_file,
            h3_resolution=8  # Resolución más alta
        )
        
        logger.info("Análisis parcial completado")
        
    except Exception as e:
        logger.error(f"Error en análisis parcial: {e}")

def run_custom_analysis():
    """Ejemplo de análisis con parámetros personalizados."""
    base_path = Path("./analisis_customizado")
    analyzer = VMSAnalyzer(base_path)
    
    try:
        # Unificar archivos de una carpeta específica
        unified_file = analyzer.unify_files(
            input_folder=base_path / "datos_custom"
        )
        
        # Filtrar por un polígono específico
        filtered_file = analyzer.filter_by_polygon(
            input_file=unified_file,
            vertices_file=base_path / "mi_poligono.csv"
        )
        
        # Análisis con configuración específica
        results = analyzer.analyze_fishing_effort(
            input_file=filtered_file,
            h3_resolution=6,  # Hexágonos más grandes
            valor_file=base_path / "valores_2023.csv"
        )
        
        logger.info("Análisis customizado completado")
        
    except Exception as e:
        logger.error(f"Error en análisis customizado: {e}")

if __name__ == "__main__":
    # Ejecutar diferentes tipos de análisis
    logger.info("Iniciando análisis completo...")
    run_full_analysis()
    
    logger.info("\nIniciando análisis parcial...")
    run_partial_analysis()
    
    logger.info("\nIniciando análisis customizado...")
    run_custom_analysis()
