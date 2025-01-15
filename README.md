# VMS Analyzer

[![PyPI version](https://badge.fury.io/py/vms-analyzer.svg)](https://badge.fury.io/py/vms-analyzer)
[![Tests](https://github.com/yourusername/vms-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/vms-analyzer/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/vms-analyzer/badge/?version=latest)](https://vms-analyzer.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Acerca de los Sistemas de Monitoreo de Embarcaciones (VMS)

Los Sistemas de Monitoreo de Embarcaciones (VMS, por sus siglas en inglés) son una herramienta fundamental en la gestión moderna de la pesca. Estos sistemas utilizan transpondedores instalados en embarcaciones pesqueras para transmitir periódicamente datos sobre:

- Posición geográfica
- Velocidad
- Rumbo
- Fecha y hora
- Identificación de la embarcación

Esta información es crucial para:
- Monitorear la actividad pesquera en tiempo real
- Verificar el cumplimiento de regulaciones pesqueras
- Proteger áreas marinas restringidas
- Analizar patrones de pesca
- Evaluar el esfuerzo pesquero

## Objetivo del Proyecto

VMS Analyzer es una biblioteca Python diseñada para transformar los datos crudos de VMS en información accionable para la gestión pesquera sostenible. Los objetivos principales son:

1. **Estandarización de Datos**: Unificar y limpiar datos VMS de diferentes fuentes y formatos.
2. **Análisis Espacial**: Identificar y analizar patrones de pesca en áreas específicas.
3. **Clasificación de Actividades**: Diferenciar automáticamente entre actividades de pesca y navegación.
4. **Evaluación de Esfuerzo**: Cuantificar el esfuerzo pesquero por áreas y períodos.
5. **Valoración Económica**: Estimar el valor económico de las actividades pesqueras por zona.

## Características Principales

### Procesamiento de Datos
- Unificación de archivos VMS de múltiples fuentes
- Estandarización automática de formatos y nombres de columnas
- Limpieza y validación de datos
- Procesamiento en paralelo para grandes volúmenes de datos

### Análisis Espacial
- Filtrado por polígonos geográficos
- Indexación espacial mediante H3
- Generación de mapas de calor de actividad pesquera
- Análisis de patrones de movimiento

### Clasificación de Actividad
- Identificación automática de actividades pesqueras
- Algoritmos basados en velocidad y patrones de movimiento
- Validación cruzada con datos históricos
- Configuración flexible de parámetros de clasificación

### Análisis de Esfuerzo Pesquero
- Cálculo de tiempo efectivo de pesca
- Análisis por embarcación, flota y área
- Integración con datos económicos
- Generación de reportes detallados

## Preparación para Machine Learning

VMS Analyzer está diseñado para facilitar el desarrollo de modelos de inteligencia artificial en el sector pesquero. La biblioteca prepara los datos para:

### 1. Modelos de Clasificación
- Identificación automática de artes de pesca
- Detección de patrones de pesca ilegal
- Clasificación de especies objetivo

### 2. Modelos Predictivos
- Predicción de zonas de pesca potenciales
- Estimación de capturas
- Pronóstico de patrones de movimiento

### 3. Análisis de Comportamiento
- Identificación de patrones anómalos
- Detección de comportamientos sospechosos
- Análisis de eficiencia operativa

## Instalación

Since this package is not yet available on PyPI, you can install it directly from GitHub:

```bash
# Create and activate conda environment
conda create -n vms_env python=3.8
conda activate vms_env

# Install dependencies
conda install -c conda-forge pandas geopandas numpy shapely tqdm matplotlib
conda install -c conda-forge h3-py contextily loguru

# Clone and install the package
git clone https://github.com/tu_usuario/vms-analyzer.git
cd vms-analyzer
pip install -e .

## Uso Básico

```python
from pathlib import Path
from vms_analyzer import VMSAnalyzer

# Inicializar analizador
analyzer = VMSAnalyzer(base_path=Path("./analysis"))

# Ejecutar análisis completo
unified_file = analyzer.unify_files()
filtered_file = analyzer.filter_by_polygon(unified_file)
classified_file = analyzer.classify_fishing_activity(filtered_file)
results = analyzer.analyze_fishing_effort(classified_file)
```

## Ejemplos de Uso

### Análisis Básico
```python
# Ejemplo de análisis básico de datos VMS
from vms_analyzer import VMSAnalyzer

analyzer = VMSAnalyzer(base_path="./vms_analysis")
results = analyzer.run_full_analysis()
```

### Análisis Personalizado
```python
# Ejemplo de análisis con parámetros personalizados
analyzer = VMSAnalyzer(
    base_path="./custom_analysis",
    chunk_size=1_000_000,
    n_cores=4
)

# Clasificar actividad con rango de velocidad específico
classified_data = analyzer.classify_fishing_activity(
    input_file="datos.csv",
    speed_range=(1, 5)
)
```

## Requisitos

- Python 3.8 o superior
- Dependencias principales:
  - pandas >= 1.3.0
  - geopandas >= 0.9.0
  - numpy >= 1.20.0
  - shapely >= 1.7.0
  - h3 >= 3.7.0
  - matplotlib >= 3.4.0
  - loguru >= 0.5.0

## Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Roadmap

### Próximas Características
- [ ] Integración con datos AIS
- [ ] Modelos pre-entrenados para clasificación de actividades
- [ ] Análisis de comportamiento basado en ML
- [ ] API REST para procesamiento en tiempo real
- [ ] Integración con bases de datos espaciales
- [ ] Herramientas de visualización interactiva

### Mejoras Planeadas
- [ ] Optimización de rendimiento para conjuntos de datos masivos
- [ ] Soporte para formatos adicionales de datos VMS
- [ ] Integración con servicios cloud
- [ ] Herramientas de validación de datos mejoradas
- [ ] Expansión de capacidades de machine learning

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Citación

Si utilizas VMS Analyzer en tu investigación, por favor cítalo como:

```bibtex
@software{vms_analyzer2025,
  author = {Ricardo Cavieses},
  title = {VMS Analyzer: A Python Library for Fishing Vessel Monitoring System Analysis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/rcavieses/vms-analyzer}
}
```

## Contacto

Ricardo Cavieses - caviesesl@uabcs.mx

Project Link: [https://github.com/rcavieses/vms-analyzer](https://github.com/rcavieses/vms-analyzer)
