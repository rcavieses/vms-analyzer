"""
Tests for the VMSAnalyzer class.
"""
import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
import shutil
import tempfile
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon

from vms_analyzer import VMSAnalyzer

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_vms_data():
    """Create sample VMS data for testing."""
    return pd.DataFrame({
        'Nombre': ['Vessel1', 'Vessel1', 'Vessel2'],
        'Razón Social': ['Company1', 'Company1', 'Company2'],
        'Fecha': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 10:30:00'],
        'Longitud': [-70.5, -70.6, -70.4],
        'Latitud': [-15.5, -15.6, -15.4],
        'Velocidad': [5.0, 3.0, 8.0]
    })

@pytest.fixture
def sample_vertices_data():
    """Create sample polygon vertices data for testing."""
    return pd.DataFrame({
        'Longitud': [-71.0, -70.0, -70.0, -71.0, -71.0],
        'Latitud': [-16.0, -16.0, -15.0, -15.0, -16.0]
    })

@pytest.fixture
def sample_valor_data():
    """Create sample economic value data for testing."""
    return pd.DataFrame({
        'VALOR': [1000000, 2000000, 3000000]
    })

@pytest.fixture
def analyzer(temp_dir):
    """Create a VMSAnalyzer instance for testing."""
    return VMSAnalyzer(base_path=temp_dir)

def test_init(analyzer, temp_dir):
    """Test VMSAnalyzer initialization."""
    assert analyzer.base_path == temp_dir
    assert analyzer.chunk_size == 5_000_000
    assert analyzer.n_cores >= 1
    
    # Check directory structure
    expected_dirs = ['input', 'unified', 'filtered', 'classified', 'analysis', 'logs']
    for dir_name in expected_dirs:
        assert (temp_dir / dir_name).exists()

def test_unify_files(analyzer, temp_dir, sample_vms_data):
    """Test file unification functionality."""
    # Create test files
    input_dir = temp_dir / 'input'
    csv_file = input_dir / 'test1.csv'
    excel_file = input_dir / 'test2.xlsx'
    
    sample_vms_data.to_csv(csv_file, index=False)
    sample_vms_data.to_excel(excel_file, index=False)
    
    # Run unification
    result_file = analyzer.unify_files()
    
    # Verify results
    assert result_file.exists()
    result_df = pd.read_csv(result_file)
    assert len(result_df) == len(sample_vms_data) * 2  # Data from both files
    assert set(result_df.columns) >= analyzer.required_columns

def test_filter_by_polygon(analyzer, temp_dir, sample_vms_data, sample_vertices_data):
    """Test polygon filtering functionality."""
    # Prepare input files
    input_file = temp_dir / 'unified' / 'vms_unified.csv'
    vertices_file = temp_dir / 'input' / 'vertices.csv'
    
    sample_vms_data.to_csv(input_file, index=False)
    sample_vertices_data.to_csv(vertices_file, index=False)
    
    # Run filtering
    result_file = analyzer.filter_by_polygon(input_file, vertices_file)
    
    # Verify results
    assert result_file.exists()
    result_df = pd.read_csv(result_file)
    assert len(result_df) <= len(sample_vms_data)  # Should only include points within polygon

def test_classify_fishing_activity(analyzer, temp_dir, sample_vms_data):
    """Test fishing activity classification."""
    # Prepare input file
    input_file = temp_dir / 'filtered' / 'vms_filtered.csv'
    sample_vms_data.to_csv(input_file, index=False)
    
    # Run classification
    result_file = analyzer.classify_fishing_activity(
        input_file=input_file,
        speed_range=(0, 6)
    )
    
    # Verify results
    assert result_file.exists()
    result_df = pd.read_csv(result_file)
    assert 'Actividad' in result_df.columns
    assert set(result_df['Actividad'].unique()) == {'Pesca', 'No Pesca'}
    
    # Verify classification logic
    assert all(result_df[result_df['Velocidad'] <= 6]['Actividad'] == 'Pesca')
    assert all(result_df[result_df['Velocidad'] > 6]['Actividad'] == 'No Pesca')

def test_analyze_fishing_effort(analyzer, temp_dir, sample_vms_data, sample_valor_data):
    """Test fishing effort analysis."""
    # Prepare input files
    input_file = temp_dir / 'classified' / 'vms_classified.csv'
    valor_file = temp_dir / 'input' / 'valor.csv'
    
    # Add classification column to sample data
    sample_vms_data['Actividad'] = 'Pesca'
    sample_vms_data.to_csv(input_file, index=False)
    sample_valor_data.to_csv(valor_file, index=False)
    
    # Run analysis
    results = analyzer.analyze_fishing_effort(
        input_file=input_file,
        h3_resolution=7,
        valor_file=valor_file
    )
    
    # Verify results
    assert isinstance(results, dict)
    assert all(Path(file).exists() for file in results.values())
    
    # Check generated files
    shp_path = results.get('shapefile')
    assert shp_path and Path(shp_path).exists()
    
    gdf = gpd.read_file(shp_path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert 'fishing_hours' in gdf.columns
    if 'val_millon' in gdf.columns:
        assert all(gdf['val_millon'] >= 0)

def test_error_handling(analyzer):
    """Test error handling for various scenarios."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        analyzer.unify_files(Path('non_existent_folder'))
    
    # Test with invalid polygon vertices
    invalid_vertices = pd.DataFrame({
        'Longitud': [-71.0],  # Incomplete polygon
        'Latitud': [-16.0]
    })
    vertices_file = analyzer.dirs['input'] / 'invalid_vertices.csv'
    invalid_vertices.to_csv(vertices_file, index=False)
    
    with pytest.raises(Exception):
        analyzer.filter_by_polygon(vertices_file=vertices_file)

def test_parallel_processing(analyzer, temp_dir, sample_vms_data):
    """Test parallel processing functionality."""
    # Create large sample dataset
    large_sample = pd.concat([sample_vms_data] * 1000, ignore_index=True)
    input_file = temp_dir / 'unified' / 'large_vms.csv'
    large_sample.to_csv(input_file, index=False)
    
    # Test with different numbers of cores
    for n_cores in [1, 2]:
        analyzer_multi = VMSAnalyzer(temp_dir, n_cores=n_cores)
        result_file = analyzer_multi.filter_by_polygon(input_file)
        assert result_file.exists()

def test_custom_parameters(temp_dir):
    """Test analyzer with custom initialization parameters."""
    custom_analyzer = VMSAnalyzer(
        base_path=temp_dir,
        chunk_size=1000,
        n_cores=1
    )
    
    assert custom_analyzer.chunk_size == 1000
    assert custom_analyzer.n_cores == 1

def test_output_consistency(analyzer, temp_dir, sample_vms_data):
    """Test consistency of output files across multiple runs."""
    # Prepare input data
    input_file = temp_dir / 'classified' / 'vms_test.csv'
    sample_vms_data.to_csv(input_file, index=False)
    
    # Run analysis multiple times
    results1 = analyzer.analyze_fishing_effort(input_file, h3_resolution=7)
    results2 = analyzer.analyze_fishing_effort(input_file, h3_resolution=7)
    
    # Compare shapefiles
    gdf1 = gpd.read_file(results1['shapefile'])
    gdf2 = gpd.read_file(results2['shapefile'])
    
    # Check if geometries and values are consistent
    assert gdf1.geometry.equals(gdf2.geometry)
    assert np.allclose(gdf1['fishing_hours'], gdf2['fishing_hours'])

if __name__ == '__main__':
    pytest.main([__file__])