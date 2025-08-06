"""
Tests for data_acquisition module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from data_acquisition import CompasDataAcquisition


class TestCompasDataAcquisition:
    """Test class for CompasDataAcquisition."""
    
    def test_init(self, temp_dir):
        """Test initialization of CompasDataAcquisition."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        assert str(acq.data_dir) == data_dir
        assert os.path.exists(acq.data_dir)
        assert acq.dataset_name == "danofer/compass"
        assert isinstance(acq.expected_files, list)
        assert len(acq.expected_files) > 0
    
    def test_check_existing_data(self, temp_dir):
        """Test checking for existing data files."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Initially no data should exist
        assert acq._check_existing_data() is False
        
        # Create a sample file
        sample_file = acq.data_dir / "compas-scores-raw.csv"
        sample_data = pd.DataFrame({
            'age': [25, 30, 35],
            'sex': ['Male', 'Female', 'Male'],
            'race': ['African-American', 'Caucasian', 'Hispanic'],
            'two_year_recid': [0, 1, 0]
        })
        sample_data.to_csv(sample_file, index=False)
        
        # Now data should exist
        assert acq._check_existing_data() is True
    
    @patch('kagglehub.dataset_download')
    def test_download_compas_data_success(self, mock_download, temp_dir):
        """Test successful COMPAS data download."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Create mock download directory with files
        mock_download_dir = temp_dir / "mock_kaggle_download"
        mock_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock CSV file
        mock_csv_path = mock_download_dir / "compas-scores-raw.csv"
        sample_data = pd.DataFrame({
            'age': [25, 30, 35],
            'sex': ['Male', 'Female', 'Male'],
            'race': ['African-American', 'Caucasian', 'Hispanic'],
            'two_year_recid': [0, 1, 0]
        })
        sample_data.to_csv(mock_csv_path, index=False)
        
        # Mock kagglehub response
        mock_download.return_value = str(mock_download_dir)
        
        # Test download
        result = acq.download_compas_data()
        
        assert result == str(acq.data_dir)
        mock_download.assert_called_once_with("danofer/compass")
        
        # Verify file was copied
        copied_file = acq.data_dir / "compas-scores-raw.csv"
        assert copied_file.exists()
    
    @patch('kagglehub.dataset_download')
    def test_download_compas_data_failure(self, mock_download, temp_dir):
        """Test failed COMPAS data download."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Mock kagglehub to raise exception
        mock_download.side_effect = Exception("Kaggle API error")
        
        # Test download failure - should raise exception
        with pytest.raises(Exception):
            acq.download_compas_data()
    
    def test_download_compas_data_existing(self, temp_dir):
        """Test download with existing data."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Create existing file
        sample_file = acq.data_dir / "compas-scores-raw.csv"
        sample_data = pd.DataFrame({
            'age': [25, 30, 35],
            'two_year_recid': [0, 1, 0]
        })
        sample_data.to_csv(sample_file, index=False)
        
        # Should return existing path without downloading
        result = acq.download_compas_data(force_reload=False)
        assert result == str(acq.data_dir)
    
    def test_load_compas_data(self, temp_dir):
        """Test loading COMPAS data."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Create sample data files
        files_data = {
            "compas-scores-raw.csv": pd.DataFrame({
                'age': [25, 30, 35],
                'sex': ['Male', 'Female', 'Male'],
                'race': ['African-American', 'Caucasian', 'Hispanic'],
                'two_year_recid': [0, 1, 0]
            }),
            "cox-violent-parsed.csv": pd.DataFrame({
                'id': [1, 2, 3],
                'violent_recid': [0, 1, 0]
            })
        }
        
        # Save files
        for filename, data in files_data.items():
            filepath = acq.data_dir / filename
            data.to_csv(filepath, index=False)
        
        # Test loading
        loaded_data = acq.load_compas_data()
        
        assert isinstance(loaded_data, dict)
        assert len(loaded_data) > 0
        
        # Check that at least one DataFrame was loaded
        for key, df in loaded_data.items():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
    
    def test_load_compas_data_no_files(self, temp_dir):
        """Test loading COMPAS data with no files available."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Should raise FileNotFoundError when no files exist
        with pytest.raises(FileNotFoundError):
            acq.load_compas_data()
    
    def test_get_dataset_info(self, temp_dir):
        """Test getting dataset information."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Create sample data file
        sample_file = acq.data_dir / "compas-scores-raw.csv"
        sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 20],
            'sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'race': ['African-American', 'Caucasian', 'Hispanic', 'Other', 'African-American'],
            'two_year_recid': [0, 1, 0, 1, 1]
        })
        sample_data.to_csv(sample_file, index=False)
        
        # Test getting info
        info = acq.get_dataset_info()
        
        assert isinstance(info, dict)
        assert len(info) > 0
        
        # Check info structure for each file
        for filename, file_info in info.items():
            assert isinstance(file_info, dict)
            assert 'shape' in file_info
            assert 'columns' in file_info
            assert 'dtypes' in file_info
    
    def test_get_available_files(self, temp_dir):
        """Test getting list of available files."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Initially no files
        available = acq.get_available_files()
        assert isinstance(available, list)
        assert len(available) == 0
        
        # Create sample files
        filenames = ["compas-scores-raw.csv", "cox-violent-parsed.csv"]
        for filename in filenames:
            filepath = acq.data_dir / filename
            pd.DataFrame({'col': [1, 2, 3]}).to_csv(filepath, index=False)
        
        # Check available files
        available = acq.get_available_files()
        assert isinstance(available, list)
        assert len(available) == 2
        
        for filename in filenames:
            assert filename in available
    
    def test_copy_files_to_data_dir(self, temp_dir):
        """Test copying files to data directory."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Create source directory with files
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        
        source_file = source_dir / "compas-scores-raw.csv"
        sample_data = pd.DataFrame({'col': [1, 2, 3]})
        sample_data.to_csv(source_file, index=False)
        
        # Test copying
        acq._copy_files_to_data_dir(str(source_dir))
        
        # Verify file was copied
        copied_file = acq.data_dir / "compas-scores-raw.csv"
        assert copied_file.exists()
        
        # Verify content is same
        copied_data = pd.read_csv(copied_file)
        pd.testing.assert_frame_equal(copied_data, sample_data)
    
    def test_validate_downloaded_files(self, temp_dir):
        """Test validation of downloaded files."""
        data_dir = str(temp_dir / "data")
        acq = CompasDataAcquisition(data_dir=data_dir)
        
        # Create valid file
        valid_file = acq.data_dir / "compas-scores-raw.csv"
        valid_data = pd.DataFrame({
            'age': [25, 30],
            'two_year_recid': [0, 1]
        })
        valid_data.to_csv(valid_file, index=False)
        
        # Should not raise exception for valid file
        try:
            acq._validate_downloaded_files()
        except Exception as e:
            pytest.fail(f"Validation failed for valid file: {e}")
        
        # Create empty file
        empty_file = acq.data_dir / "empty.csv"
        empty_file.touch()
        
        # Should handle empty files gracefully
        try:
            acq._validate_downloaded_files()
        except Exception:
            # Expected behavior - validation might fail for empty files
            pass


# Test standalone functions
def test_standalone_download_compas_data(temp_dir):
    """Test standalone download_compas_data function."""
    from data_acquisition import download_compas_data
    
    data_dir = str(temp_dir / "data")
    
    # Create existing file to avoid actual download
    os.makedirs(data_dir, exist_ok=True)
    sample_file = Path(data_dir) / "compas-scores-raw.csv"
    pd.DataFrame({'col': [1, 2, 3]}).to_csv(sample_file, index=False)
    
    # Test function
    result = download_compas_data(data_dir=data_dir, force_reload=False)
    assert result == data_dir
    assert os.path.exists(sample_file)


def test_standalone_load_compas_data(temp_dir):
    """Test standalone load_compas_data function."""
    from data_acquisition import load_compas_data
    
    data_dir = str(temp_dir / "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample file
    sample_file = Path(data_dir) / "compas-scores-raw.csv"
    sample_data = pd.DataFrame({
        'age': [25, 30, 35],
        'two_year_recid': [0, 1, 0]
    })
    sample_data.to_csv(sample_file, index=False)
    
    # Test function
    result = load_compas_data(data_dir=data_dir)
    assert isinstance(result, dict)
    assert len(result) > 0
    
    for key, df in result.items():
        assert isinstance(df, pd.DataFrame)


def test_standalone_get_dataset_info(temp_dir):
    """Test standalone get_dataset_info function."""
    from data_acquisition import get_dataset_info
    
    data_dir = str(temp_dir / "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample file
    sample_file = Path(data_dir) / "compas-scores-raw.csv"
    sample_data = pd.DataFrame({
        'age': [25, 30, 35],
        'two_year_recid': [0, 1, 0]
    })
    sample_data.to_csv(sample_file, index=False)
    
    # Test function
    result = get_dataset_info(data_dir=data_dir)
    assert isinstance(result, dict)
    assert len(result) > 0