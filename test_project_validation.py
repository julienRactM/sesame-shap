#!/usr/bin/env python3
"""
Final project validation script for COMPAS SHAP Analysis.

This script performs a comprehensive validation of the entire project,
testing imports, basic functionality, and generating a validation report.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

class ProjectValidator:
    """Comprehensive project validation."""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_result(self, test_name, success, message=""):
        """Log test result."""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        self.results[test_name] = {
            'success': success,
            'message': message,
            'status': status
        }
        
        print(f"{status}: {test_name}")
        if message and not success:
            print(f"      └─ {message}")
    
    def test_imports(self):
        """Test all module imports."""
        print("\n🔍 Testing Module Imports...")
        
        modules_to_test = [
            'data_acquisition',
            'exploratory_analysis', 
            'feature_engineering',
            'model_training',
            'shap_analysis',
            'bias_analysis',
            'bias_mitigation',
            'fairness_evaluation',
            'interpretability_comparison'
        ]
        
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                self.log_result(f"Import {module_name}", True)
            except Exception as e:
                self.log_result(f"Import {module_name}", False, str(e))
    
    def test_class_instantiation(self):
        """Test instantiation of main classes."""
        print("\n🔍 Testing Class Instantiation...")
        
        # Test data acquisition
        try:
            from data_acquisition import CompasDataAcquisition
            acq = CompasDataAcquisition(data_dir="data/raw")
            self.log_result("CompasDataAcquisition instantiation", True)
        except Exception as e:
            self.log_result("CompasDataAcquisition instantiation", False, str(e))
        
        # Test feature engineering
        try:
            from feature_engineering import COMPASFeatureEngineer
            fe = COMPASFeatureEngineer()
            self.log_result("COMPASFeatureEngineer instantiation", True)
        except Exception as e:
            self.log_result("COMPASFeatureEngineer instantiation", False, str(e))
        
        # Test model training
        try:
            from model_training import CompasModelTrainer
            trainer = CompasModelTrainer()
            self.log_result("CompasModelTrainer instantiation", True)
        except Exception as e:
            self.log_result("CompasModelTrainer instantiation", False, str(e))
        
        # Test SHAP analysis
        try:
            from shap_analysis import CompasShapAnalyzer
            shap_analyzer = CompasShapAnalyzer()
            self.log_result("CompasShapAnalyzer instantiation", True)
        except Exception as e:
            self.log_result("CompasShapAnalyzer instantiation", False, str(e))
        
        # Test bias analysis
        try:
            from bias_analysis import CompasBiasAnalyzer
            bias_analyzer = CompasBiasAnalyzer()
            self.log_result("CompasBiasAnalyzer instantiation", True)
        except Exception as e:
            self.log_result("CompasBiasAnalyzer instantiation", False, str(e))
        
        # Test bias mitigation
        try:
            from bias_mitigation import BiasMitigationFramework
            mitigator = BiasMitigationFramework()
            self.log_result("BiasMitigationFramework instantiation", True)
        except Exception as e:
            self.log_result("BiasMitigationFramework instantiation", False, str(e))
    
    def test_dependencies(self):
        """Test critical dependencies."""
        print("\n🔍 Testing Critical Dependencies...")
        
        dependencies = [
            ('pandas', 'pd'),
            ('numpy', 'np'),
            ('matplotlib.pyplot', 'plt'),
            ('seaborn', 'sns'),
            ('sklearn.ensemble', 'RandomForestClassifier'),
            ('sklearn.linear_model', 'LogisticRegression'),
            ('sklearn.metrics', 'accuracy_score'),
            ('shap', None),
            ('plotly.graph_objects', 'go'),
            ('plotly.express', 'px'),
            ('fairlearn.metrics', None),
            ('imblearn.over_sampling', 'SMOTE'),
        ]
        
        for dep in dependencies:
            module_name = dep[0]
            import_as = dep[1]
            
            try:
                if import_as:
                    if '.' in import_as:
                        # For specific imports like RandomForestClassifier
                        module = importlib.import_module(module_name)
                        getattr(module, import_as)
                    else:
                        # For aliased imports like 'pd', 'np'
                        importlib.import_module(module_name)
                else:
                    # For simple imports
                    importlib.import_module(module_name)
                
                self.log_result(f"Dependency {module_name}", True)
            except Exception as e:
                self.log_result(f"Dependency {module_name}", False, str(e))
    
    def test_file_structure(self):
        """Test project file structure."""
        print("\n🔍 Testing Project Structure...")
        
        required_files = [
            "README.md",
            "CLAUDE.md",
            "requirements.txt",
            "install.sh",
            "main_notebook.ipynb",
            "src/__init__.py",
            "src/data_acquisition.py",
            "src/exploratory_analysis.py",
            "src/feature_engineering.py",
            "src/model_training.py",
            "src/shap_analysis.py",
            "src/bias_analysis.py",
            "src/bias_mitigation.py",
            "src/fairness_evaluation.py",
            "src/interpretability_comparison.py",
            "Dashboard/app.py"
        ]
        
        required_dirs = [
            "data",
            "data/raw",
            "data/processed", 
            "data/models",
            "data/results",
            "src",
            "Dashboard",
            "tests"
        ]
        
        # Test files
        for file_path in required_files:
            full_path = project_root / file_path
            exists = full_path.exists()
            self.log_result(f"File {file_path}", exists, 
                           "" if exists else "File not found")
        
        # Test directories
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            self.log_result(f"Directory {dir_path}", exists,
                           "" if exists else "Directory not found")
    
    def test_data_availability(self):
        """Test if sample data is available."""
        print("\n🔍 Testing Data Availability...")
        
        try:
            from data_acquisition import CompasDataAcquisition
            acq = CompasDataAcquisition()
            
            # Try to load some data (will create sample if needed)
            try:
                data_dict = acq.load_compas_data()
                has_data = len(data_dict) > 0
                self.log_result("COMPAS data loading", has_data,
                               "" if has_data else "No data available")
                
                if has_data:
                    # Check data quality
                    first_df = next(iter(data_dict.values()))
                    has_records = len(first_df) > 0
                    self.log_result("Data has records", has_records,
                                   f"Found {len(first_df)} records" if has_records else "No records")
                    
            except Exception as e:
                self.log_result("COMPAS data loading", False, str(e))
                
        except Exception as e:
            self.log_result("Data acquisition setup", False, str(e))
    
    def test_notebook_structure(self):
        """Test notebook structure."""
        print("\n🔍 Testing Notebook...")
        
        notebook_path = project_root / "main_notebook.ipynb"
        if notebook_path.exists():
            try:
                import json
                with open(notebook_path, 'r') as f:
                    notebook = json.load(f)
                
                has_cells = 'cells' in notebook and len(notebook['cells']) > 0
                self.log_result("Notebook has cells", has_cells,
                               f"Found {len(notebook.get('cells', []))} cells" if has_cells else "No cells found")
                
                # Check for key sections
                notebook_content = str(notebook).lower()
                sections = [
                    "shap",
                    "bias", 
                    "compas",
                    "model",
                    "data"
                ]
                
                for section in sections:
                    has_section = section in notebook_content
                    self.log_result(f"Notebook mentions {section}", has_section)
                
            except Exception as e:
                self.log_result("Notebook parsing", False, str(e))
        else:
            self.log_result("Notebook exists", False, "main_notebook.ipynb not found")
    
    def run_validation(self):
        """Run complete validation."""
        print("🚀 COMPAS SHAP Project Validation")
        print("=" * 50)
        print(f"Validation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        self.test_file_structure()
        self.test_dependencies()
        self.test_imports()
        self.test_class_instantiation()
        self.test_data_availability()
        self.test_notebook_structure()
        
        # Generate report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final validation report."""
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"\n📈 RESULTS:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   ✅ Passed: {self.passed_tests}")
        print(f"   ❌ Failed: {self.total_tests - self.passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Categorize results
        categories = {
            'File Structure': [],
            'Dependencies': [],
            'Module Imports': [],
            'Class Instantiation': [],
            'Data': [],
            'Notebook': []
        }
        
        for test_name, result in self.results.items():
            if 'File' in test_name or 'Directory' in test_name:
                categories['File Structure'].append((test_name, result))
            elif 'Dependency' in test_name:
                categories['Dependencies'].append((test_name, result))
            elif 'Import' in test_name:
                categories['Module Imports'].append((test_name, result))
            elif 'instantiation' in test_name:
                categories['Class Instantiation'].append((test_name, result))
            elif 'data' in test_name.lower() or 'Data' in test_name:
                categories['Data'].append((test_name, result))
            elif 'Notebook' in test_name:
                categories['Notebook'].append((test_name, result))
        
        print(f"\n📋 DETAILED RESULTS BY CATEGORY:")
        for category, tests in categories.items():
            if tests:
                passed = sum(1 for _, result in tests if result['success'])
                total = len(tests)
                print(f"\n   {category}: {passed}/{total} passed")
                for test_name, result in tests:
                    status_icon = "✅" if result['success'] else "❌"
                    print(f"      {status_icon} {test_name}")
                    if not result['success'] and result['message']:
                        print(f"          └─ {result['message']}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if success_rate >= 90:
            print("   🎉 EXCELLENT - Project is ready for production use!")
        elif success_rate >= 75:
            print("   ✅ GOOD - Project is functional with minor issues")
        elif success_rate >= 50:
            print("   ⚠️  FAIR - Project has significant issues that need attention")
        else:
            print("   ❌ POOR - Project has major issues that must be fixed")
        
        # Save detailed report
        report_path = project_root / "validation_report.txt"
        with open(report_path, 'w') as f:
            f.write("COMPAS SHAP Project Validation Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Success Rate: {success_rate:.1f}% ({self.passed_tests}/{self.total_tests})\n\n")
            
            for test_name, result in self.results.items():
                f.write(f"{result['status']}: {test_name}\n")
                if result['message']:
                    f.write(f"    {result['message']}\n")
                f.write("\n")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Recommendations
        if success_rate < 100:
            print(f"\n💡 NEXT STEPS:")
            failed_tests = [name for name, result in self.results.items() if not result['success']]
            if failed_tests:
                print("   1. Fix the failed tests listed above")
                print("   2. Run the validation again to verify fixes")
                if any('Dependency' in test for test in failed_tests):
                    print("   3. Install missing dependencies with: pip install -r requirements.txt")
                if any('Import' in test for test in failed_tests):
                    print("   4. Check module implementations for syntax errors")


def main():
    """Main validation function."""
    validator = ProjectValidator()
    validator.run_validation()


if __name__ == "__main__":
    main()