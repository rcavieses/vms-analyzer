"""
Tests for utility functions in the VMS Analyzer library.
"""
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from datetime import datetime

from vms_analyzer import utils

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_hex_times():
    """Create sample hexagon time data."""
    return pd.DataFrame({
        'h3_index': ['8928308280fffff', '8928308281fffff', '8928308282fffff'],
        'fishing_hours': [10.5, 15.2, 8.7]
    })

@pytest.fixture
def sample_vms_data():
    """Create sample VMS tracking data."""
    return pd.DataFrame({
        'Nombre': ['Vessel1', 'Vessel1', 'Vessel2'],
        'Fecha': pd.date_range('2024-01-01', periods=3, freq='H'),
        'Longitud': [-70.5, -70.6, -70.4],
        'Latitud': [-15.5, -15.6, -15.4],
        'h3_index': ['8928308280fffff', '8928308280fffff', '8928308281fffff']
    })

def test_setup_directories(temp_dir):
    """Test directory setup functionality."""
    # Test directory creation
    dirs = utils.setup_directories(temp_dir)
    
    # Verify all expected directories exist
    expected_dirs = ['input', 'unified', 'filtered', 'classified', 'analysis', 'logs']
    for dir_name in expected_dirs:
        assert dirs[dir_name].exists()
        assert dirs[dir_name].is_dir()
    
    # Test idempotency (running twice shouldn't cause errors)
    dirs_again = utils.setup_directories(temp_dir)
    assert dirs == dirs_again

def test_process_single_file(temp_dir):
    """Test processing of individual VMS data files."""
    # Create test file
    test_data = pd.DataFrame({
        'Embarcación': ['Vessel1'],
        'Razón_Social': ['Company1'],
        'Fecha': ['2024-01-01 10:00:00'],
        'Longitud': [-70.5],
        'Latitud': [-15.5],
        'Velocidad': [5.0]
    })
    
    # Test CSV processing
    csv_file = temp_dir / 'test.csv'
    test_data.to_csv(csv_file, index=False)
    
    column_mapping = {
        'Embarcación': 'Nombre',
        'Razón_Social': 'Razón Social'
    }
    required_columns = {'Nombre', 'Razón Social', 'Fecha', 'Longitud', 'Latitud'}
    
    result = utils.process_single_file(csv_file, column_mapping, required_columns)
    assert isinstance(result, pd.DataFrame)
    assert 'Nombre' in result.columns
    assert len(result) == 1
    
    # Test Excel processing
    excel_file = temp_dir / 'test.xlsx'
    test_data.to_excel(excel_file, index=False)
    result_excel = utils.process_single_file(excel_file, column_mapping, required_columns)
    assert isinstance(result_excel, pd.DataFrame)
    
    # Test error handling with invalid file
    invalid_file = temp_dir / 'nonexistent.csv'
    result_invalid = utils.process_single_file(invalid_file, column_mapping, required_columns)
    assert result_invalid is None

def test_process_polygon_chunk():
    """Test processing of data chunks for polygon filtering."""
    # Create sample data
    chunk = pd.DataFrame({
        'Longitud': [-70.5, -70.6, -70.4],
        'Latitud': [-15.5, -15.6, -15.4],
        'Valor': [1, 2, 3]
    })
    
    # Create test polygon that includes all points
    polygon = Polygon([
        (-71, -16), (-70, -16), (-70, -15), (-71, -15), (-71, -16)
    ])
    
    # Test processing
    result = utils.process_polygon_chunk((chunk, polygon))
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 3  # All points should be inside
    assert 'geometry' in result.columns
    
    # Test with polygon that excludes all points
    outside_polygon = Polygon([
        (-72, -17), (-71.5, -17), (-71.5, -16.5), (-72, -16.5), (-72, -17)
    ])
    result_outside = utils.process_polygon_chunk((chunk, outside_polygon))
    assert len(result_outside) == 0

def test_calculate_hex_times(sample_vms_data):
    """Test calculation of fishing time per hexagon."""
    result = utils.calculate_hex_times(sample_vms_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'h3_index' in result.columns
    assert 'fishing_hours' in result.columns
    
    # Test time calculation
    assert len(result) <= len(sample_vms_data['h3_index'].unique())
    assert all(result['fishing_hours'] >= 0)
    
    # Test with custom max_gap
    result_custom = utils.calculate_hex_times(sample_vms_data, max_gap=1.0)
    assert isinstance(result_custom, pd.DataFrame)

def test_create_hex_gdf(sample_hex_times):
    """Test creation of GeoDataFrame with H3 hexagons."""
    result = utils.create_hex_gdf(sample_hex_times, resolution=8)
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert 'geometry' in result.columns
    assert result.crs == "EPSG:4326"
    assert len(result) == len(sample_hex_times)
    
    # Verify geometries
    assert all(isinstance(geom, Polygon) for geom in result.geometry)

def test_create_effort_map(temp_dir, sample_hex_times):
    """Test creation of fishing effort map."""
    # Create GeoDataFrame for testing
    hex_gdf = utils.create_hex_gdf(sample_hex_times, resolution=8)
    output_path = temp_dir / 'test_map.png'
    
    # Test linear scale map
    utils.create_effort_map(hex_gdf, output_path, scale='linear')
    assert output_path.exists()
    
    # Test logarithmic scale map
    log_path = temp_dir / 'test_map_log.png'
    utils.create_effort_map(hex_gdf, log_path, scale='log')
    assert log_path.exists()
    
    # Test with custom title
    custom_path = temp_dir / 'test_map_custom.png'
    utils.create_effort_map(hex_gdf, custom_path, title='Custom Map Title')
    assert custom_path.exists()
    
    # Test error handling
    with pytest.raises(Exception):
        utils.create_effort_map(None, output_path)

def test_error_handling():
    """Test error handling in utility functions."""
    # Test with invalid dataframe
    invalid_df = pd.DataFrame()
    with pytest.raises(Exception):
        utils.calculate_hex_times(invalid_df)
    
    # Test with invalid H3 indexes
    invalid_hex_times = pd.DataFrame({
        'h3_index': ['invalid_index'],
        'fishing_hours': [10.5]
    })
    with pytest.raises(Exception):
        utils.create_hex_gdf(invalid_hex_times, resolution=8)

def test_edge_cases(sample_vms_data):
    """Test edge cases and boundary conditions."""
    # Test with empty dataframe
    empty_df = sample_vms_data[0:0]  # Empty DataFrame with same columns
    result_empty = utils.calculate_hex_times(empty_df)
    assert len(result_empty) == 0
    
    # Test with single row
    single_row = sample_vms_data.iloc[0:1]
    result_single = utils.calculate_hex_times(single_row)
    assert len(result_single) <= 1
    
    # Test with duplicate indexes
    duplicate_data = pd.concat([sample_vms_data, sample_vms_data])
    result_duplicate = utils.calculate_hex_times(duplicate_data)
    assert len(result_duplicate) <= len(duplicate_data['h3_index'].unique())

if __name__ == '__main__':
    pytest.main([__file__])
