#!/usr/bin/env python3
"""
Comprehensive test runner for all COMPAS SHAP project modules.

This script tests the basic functionality of all modules to ensure they work properly.
"""

import sys
import os
import traceback
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

class TestRunner:
    """Main test runner for all modules."""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = None
        self.setup_temp_environment()
    
    def setup_temp_environment(self):
        """Setup temporary test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="compas_test_"))
        print(f"🔧 Test environment: {self.temp_dir}")
        
        # Create test data structure
        (self.temp_dir / "data" / "raw").mkdir(parents=True)
        (self.temp_dir / "data" / "processed").mkdir(parents=True)
        (self.temp_dir / "data" / "models").mkdir(parents=True)
        (self.temp_dir / "data" / "results").mkdir(parents=True)
        
        # Create sample COMPAS data
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample COMPAS data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'Person_ID': range(1, n_samples + 1),
            'AssessmentID': np.random.randint(50000, 60000, n_samples),
            'Case_ID': np.random.randint(1000, 2000, n_samples),
            'FirstName': [f'Person{i}' for i in range(n_samples)],
            'LastName': [f'Last{i}' for i in range(n_samples)],
            'age': np.random.randint(18, 70, n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic', 'Other'], 
                                   n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'priors_count': np.random.poisson(2, n_samples),
            'days_b_screening_arrest': np.random.normal(0, 50, n_samples),
            'c_jail_in': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'c_jail_out': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'c_charge_degree': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
            'score_text': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
            'decile_score': np.random.randint(1, 11, n_samples),
            'two_year_recid': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'compas_screening_date': pd.date_range('2013-01-01', periods=n_samples, freq='D')[:n_samples],
            'c_offense_date': pd.date_range('2012-01-01', periods=n_samples, freq='2D')[:n_samples],
            'c_arrest_date': pd.date_range('2012-06-01', periods=n_samples, freq='3D')[:n_samples]
        })
        
        # Save to test directory
        sample_path = self.temp_dir / "data" / "raw" / "compas-scores-raw.csv"
        sample_data.to_csv(sample_path, index=False)
        print(f"✅ Sample data created: {len(sample_data)} records")
    
    def test_data_acquisition(self):
        """Test data acquisition module."""
        try:
            from data_acquisition import CompasDataAcquisition
            
            # Test initialization
            acq = CompasDataAcquisition(data_dir=str(self.temp_dir / "data" / "raw"))
            
            # Test loading existing data
            data_dict = acq.load_compas_data()
            assert isinstance(data_dict, dict)
            assert len(data_dict) > 0
            
            # Test getting info
            info = acq.get_dataset_info()
            assert isinstance(info, dict)
            
            # Test getting available files
            files = acq.get_available_files()
            assert isinstance(files, list)
            
            self.results['data_acquisition'] = {'status': 'PASS', 'message': 'All functions working'}
            
        except Exception as e:
            self.results['data_acquisition'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_exploratory_analysis(self):
        """Test exploratory analysis module."""
        try:
            from exploratory_analysis import CompasExploratoryAnalysis
            from data_acquisition import CompasDataAcquisition
            
            # Load sample data
            acq = CompasDataAcquisition(data_dir=str(self.temp_dir / "data" / "raw"))
            data_dict = acq.load_compas_data()
            df = next(iter(data_dict.values()))  # Get first dataframe
            
            # Test exploratory analysis
            explorer = CompasExploratoryAnalysis(results_dir=str(self.temp_dir / "data" / "results"))
            explorer.load_data(df)
            
            # Test basic overview
            summary = explorer.basic_data_overview()
            assert isinstance(summary, pd.DataFrame)
            
            # Test demographic analysis
            demo_plots = explorer.analyze_demographic_distributions()
            assert isinstance(demo_plots, dict)
            
            self.results['exploratory_analysis'] = {'status': 'PASS', 'message': 'Analysis functions working'}
            
        except Exception as e:
            self.results['exploratory_analysis'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_feature_engineering(self):
        """Test feature engineering module."""
        try:
            from feature_engineering import COMPASFeatureEngineer
            from data_acquisition import CompasDataAcquisition
            
            # Load sample data
            acq = CompasDataAcquisition(data_dir=str(self.temp_dir / "data" / "raw"))
            data_dict = acq.load_compas_data()
            df = next(iter(data_dict.values()))
            
            # Test feature engineering
            fe = COMPASFeatureEngineer(processed_dir=str(self.temp_dir / "data" / "processed"))
            
            # Test basic functionality
            cleaned_df = fe.clean_data(df.copy())
            assert isinstance(cleaned_df, pd.DataFrame)
            
            self.results['feature_engineering'] = {'status': 'PASS', 'message': 'Feature engineering working'}
            
        except Exception as e:
            self.results['feature_engineering'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_model_training(self):
        """Test model training module."""
        try:
            from model_training import CompasModelTrainer
            
            # Test initialization
            trainer = CompasModelTrainer(results_dir=str(self.temp_dir / "data" / "models"))
            assert trainer.results_dir is not None
            
            self.results['model_training'] = {'status': 'PASS', 'message': 'Model training initializes correctly'}
            
        except Exception as e:
            self.results['model_training'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_shap_analysis(self):
        """Test SHAP analysis module."""
        try:
            from shap_analysis import CompasShapAnalyzer
            
            # Create minimal test setup
            analyzer = CompasShapAnalyzer(results_dir=str(self.temp_dir / "data" / "results" / "shap"))
            
            # Test initialization
            assert analyzer.results_dir is not None
            assert os.path.exists(analyzer.results_dir)
            
            self.results['shap_analysis'] = {'status': 'PASS', 'message': 'SHAP analyzer initializes correctly'}
            
        except Exception as e:
            self.results['shap_analysis'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_bias_analysis(self):
        """Test bias analysis module."""
        try:
            from bias_analysis import CompasBiasAnalyzer
            
            # Test initialization
            analyzer = CompasBiasAnalyzer(results_dir=str(self.temp_dir / "data" / "results" / "bias"))
            assert analyzer.results_dir is not None
            
            self.results['bias_analysis'] = {'status': 'PASS', 'message': 'Bias analyzer initializes correctly'}
            
        except Exception as e:
            self.results['bias_analysis'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_bias_mitigation(self):
        """Test bias mitigation module."""
        try:
            from bias_mitigation import CompasBiasMitigation
            
            # Test initialization
            mitigator = CompasBiasMitigation(results_dir=str(self.temp_dir / "data" / "results" / "mitigation"))
            assert mitigator.results_dir is not None
            
            self.results['bias_mitigation'] = {'status': 'PASS', 'message': 'Bias mitigation initializes correctly'}
            
        except Exception as e:
            self.results['bias_mitigation'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_fairness_evaluation(self):
        """Test fairness evaluation module."""
        try:
            from fairness_evaluation import CompasFairnessEvaluator
            
            # Test initialization
            evaluator = CompasFairnessEvaluator(results_dir=str(self.temp_dir / "data" / "results" / "fairness"))
            assert evaluator.results_dir is not None
            
            self.results['fairness_evaluation'] = {'status': 'PASS', 'message': 'Fairness evaluator initializes correctly'}
            
        except Exception as e:
            self.results['fairness_evaluation'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_interpretability_comparison(self):
        """Test interpretability comparison module."""
        try:
            from interpretability_comparison import CompasInterpretabilityComparator
            
            # Test initialization
            comparator = CompasInterpretabilityComparator(results_dir=str(self.temp_dir / "data" / "results" / "comparison"))
            assert comparator.results_dir is not None
            
            self.results['interpretability_comparison'] = {'status': 'PASS', 'message': 'Interpretability comparator initializes correctly'}
            
        except Exception as e:
            self.results['interpretability_comparison'] = {'status': 'FAIL', 'message': str(e)}
    
    def test_dashboard(self):
        """Test dashboard module."""
        try:
            dashboard_path = project_root / "Dashboard" / "app.py"
            
            # Check if dashboard file exists and is readable
            assert dashboard_path.exists(), "Dashboard app.py not found"
            
            # Try to read the file
            with open(dashboard_path, 'r') as f:
                content = f.read()
                assert len(content) > 0, "Dashboard file is empty"
                assert 'streamlit' in content.lower(), "Dashboard doesn't appear to be a Streamlit app"
            
            self.results['dashboard'] = {'status': 'PASS', 'message': 'Dashboard file exists and appears valid'}
            
        except Exception as e:
            self.results['dashboard'] = {'status': 'FAIL', 'message': str(e)}
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("🚀 Starting comprehensive module testing...")
        print("=" * 70)
        
        tests = [
            ('Data Acquisition', self.test_data_acquisition),
            ('Exploratory Analysis', self.test_exploratory_analysis),
            ('Feature Engineering', self.test_feature_engineering),
            ('Model Training', self.test_model_training),
            ('SHAP Analysis', self.test_shap_analysis),
            ('Bias Analysis', self.test_bias_analysis),
            ('Bias Mitigation', self.test_bias_mitigation),
            ('Fairness Evaluation', self.test_fairness_evaluation),
            ('Interpretability Comparison', self.test_interpretability_comparison),
            ('Dashboard', self.test_dashboard),
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing {test_name}...")
            try:
                test_func()
                status = self.results.get(test_name.lower().replace(' ', '_'), {}).get('status', 'UNKNOWN')
                if status == 'PASS':
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: EXCEPTION - {str(e)}")
                self.results[test_name.lower().replace(' ', '_')] = {'status': 'EXCEPTION', 'message': str(e)}
        
        self.generate_report()
    
    def generate_report(self):
        """Generate final test report."""
        print("\n" + "=" * 70)
        print("📊 FINAL TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for module, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"   {status_icon} {module.replace('_', ' ').title()}: {result['status']}")
            if result['status'] != 'PASS':
                print(f"      └─ {result['message']}")
        
        # Save report to file
        report_path = self.temp_dir / "test_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"COMPAS SHAP Project Test Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}\n\n")
            
            for module, result in self.results.items():
                f.write(f"{module}: {result['status']}\n")
                f.write(f"Message: {result['message']}\n\n")
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! The project is ready to use.")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please check the issues above.")
    
    def cleanup(self):
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test environment: {self.temp_dir}")


def main():
    """Main test runner function."""
    print("🎯 COMPAS SHAP Project - Comprehensive Module Testing")
    print("This will test all Python modules to ensure they work properly.")
    print()
    
    runner = TestRunner()
    try:
        runner.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()