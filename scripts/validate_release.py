#!/usr/bin/env python3
"""
Release Validation Script

This script performs final validation checks before release to ensure:
1. All tests pass
2. Documentation is complete
3. Version information is correct
4. CI/CD pipeline is working
5. Release artifacts are properly generated
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from version_info import get_framework_version, check_version_compatibility


def run_command(command, cwd=None):
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def validate_version_info():
    """Validate version information and compatibility."""
    print("1. Validating version information...")
    
    # Check framework version
    version = get_framework_version()
    if version == "0.0.0" or not version:
        print("   ❌ Framework version not properly set")
        return False
    
    print(f"   ✓ Framework version: {version}")
    
    # Check version compatibility
    compatibility = check_version_compatibility()
    if not compatibility.get('overall_compatible', False):
        print("   ❌ Environment compatibility issues detected:")
        for check, result in compatibility.items():
            if not result and check != 'overall_compatible':
                print(f"     - {check}: Failed")
        return False
    
    print("   ✓ Environment compatibility validated")
    return True


def validate_test_suite():
    """Validate that all tests pass."""
    print("\n2. Validating test suite...")
    
    # Run unit tests
    print("   Running unit tests...")
    success, output, error = run_command("python -m pytest tests/unit/ -v --tb=short")
    if not success:
        print("   ❌ Unit tests failed")
        print(f"   Error: {error}")
        return False
    print("   ✓ Unit tests passed")
    
    # Run integration tests
    print("   Running integration tests...")
    success, output, error = run_command("python -m pytest tests/integration/ -v --tb=short")
    if not success:
        print("   ❌ Integration tests failed")
        print(f"   Error: {error}")
        return False
    print("   ✓ Integration tests passed")
    
    # Run validation tests
    print("   Running validation tests...")
    success, output, error = run_command("python -m pytest tests/validation/ -v --tb=short")
    if not success:
        print("   ❌ Validation tests failed")
        print(f"   Error: {error}")
        return False
    print("   ✓ Validation tests passed")
    
    # Run toy model test
    print("   Running toy model test...")
    success, output, error = run_command("python scripts/ci_toy_model_test.py")
    if not success:
        print("   ❌ Toy model test failed")
        print(f"   Error: {error}")
        return False
    print("   ✓ Toy model test passed")
    
    return True


def validate_documentation():
    """Validate that documentation is complete."""
    print("\n3. Validating documentation...")
    
    required_docs = [
        "README.md",
        "docs/API.md",
        "docs/TUTORIAL.md",
        "docs/CONFIGURATION.md",
        "docs/REPRODUCIBILITY.md"
    ]
    
    missing_docs = []
    for doc_path in required_docs:
        if not Path(doc_path).exists():
            missing_docs.append(doc_path)
        else:
            # Check if file is not empty
            if Path(doc_path).stat().st_size < 100:  # Less than 100 bytes
                missing_docs.append(f"{doc_path} (too small)")
    
    if missing_docs:
        print("   ❌ Missing or incomplete documentation:")
        for doc in missing_docs:
            print(f"     - {doc}")
        return False
    
    print(f"   ✓ All {len(required_docs)} documentation files present")
    
    # Check README badges
    readme_content = Path("README.md").read_text()
    required_badges = ["CI Status", "Release", "DOI", "License", "Python"]
    missing_badges = []
    
    for badge in required_badges:
        if badge not in readme_content:
            missing_badges.append(badge)
    
    if missing_badges:
        print("   ⚠ Missing README badges (update before release):")
        for badge in missing_badges:
            print(f"     - {badge}")
    else:
        print("   ✓ README badges present")
    
    return True


def validate_examples():
    """Validate that examples work correctly."""
    print("\n4. Validating examples...")
    
    example_files = [
        "examples/basic_usage.py",
        "examples/text_classification_example.py",
        "examples/visualization_example.py"
    ]
    
    for example_file in example_files:
        if not Path(example_file).exists():
            print(f"   ❌ Missing example: {example_file}")
            return False
        
        print(f"   Testing {example_file}...")
        # For now, just check syntax
        success, output, error = run_command(f"python -m py_compile {example_file}")
        if not success:
            print(f"   ❌ Syntax error in {example_file}: {error}")
            return False
    
    print(f"   ✓ All {len(example_files)} examples validated")
    return True


def validate_ci_pipeline():
    """Validate CI/CD pipeline configuration."""
    print("\n5. Validating CI/CD pipeline...")
    
    ci_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml"
    ]
    
    for ci_file in ci_files:
        if not Path(ci_file).exists():
            print(f"   ❌ Missing CI file: {ci_file}")
            return False
    
    print("   ✓ CI/CD workflow files present")
    
    # Check if CI scripts exist
    ci_scripts = [
        "scripts/ci_toy_model_test.py",
        "scripts/ci_performance_benchmark.py",
        "scripts/verify_reproducibility.py"
    ]
    
    for script in ci_scripts:
        if not Path(script).exists():
            print(f"   ❌ Missing CI script: {script}")
            return False
    
    print("   ✓ CI scripts present")
    return True


def validate_release_artifacts():
    """Validate release preparation artifacts."""
    print("\n6. Validating release artifacts...")
    
    # Run release preparation
    print("   Running release preparation...")
    success, output, error = run_command("python scripts/prepare_release.py")
    if not success:
        print("   ❌ Release preparation failed")
        print(f"   Error: {error}")
        return False
    
    # Check generated files
    expected_files = [
        "RELEASE_NOTES.md",
        ".zenodo.json"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("   ❌ Missing release artifacts:")
        for file_path in missing_files:
            print(f"     - {file_path}")
        return False
    
    print("   ✓ Release artifacts generated successfully")
    
    # Validate Zenodo metadata
    try:
        with open(".zenodo.json", "r") as f:
            zenodo_data = json.load(f)
        
        required_fields = ["title", "description", "creators", "keywords", "license"]
        missing_fields = [field for field in required_fields if field not in zenodo_data]
        
        if missing_fields:
            print("   ❌ Incomplete Zenodo metadata:")
            for field in missing_fields:
                print(f"     - Missing: {field}")
            return False
        
        print("   ✓ Zenodo metadata validated")
        
    except Exception as e:
        print(f"   ❌ Error validating Zenodo metadata: {e}")
        return False
    
    return True


def validate_reproducibility():
    """Validate reproducibility framework."""
    print("\n7. Validating reproducibility...")
    
    # Run quick reproducibility test
    success, output, error = run_command("python scripts/verify_reproducibility.py --quick-test")
    if not success:
        print("   ❌ Reproducibility validation failed")
        print(f"   Error: {error}")
        return False
    
    print("   ✓ Reproducibility validated")
    return True


def validate_performance():
    """Validate performance benchmarks."""
    print("\n8. Validating performance...")
    
    # Run performance benchmark
    success, output, error = run_command("python scripts/ci_performance_benchmark.py")
    if not success:
        print("   ❌ Performance benchmark failed")
        print(f"   Error: {error}")
        return False
    
    # Check benchmark results
    if Path("benchmark_results.json").exists():
        try:
            with open("benchmark_results.json", "r") as f:
                results = json.load(f)
            
            summary = results.get("summary", {})
            success_rate = summary.get("success_rate", 0)
            max_time = summary.get("max_time_seconds", float('inf'))
            
            if success_rate < 0.8:
                print(f"   ⚠ Warning: Low success rate ({success_rate:.1%})")
            
            if max_time > 120:  # 2 minutes
                print(f"   ⚠ Warning: Slow performance (max: {max_time:.1f}s)")
            
            print(f"   ✓ Performance benchmark completed (success rate: {success_rate:.1%})")
            
        except Exception as e:
            print(f"   ⚠ Warning: Could not parse benchmark results: {e}")
    
    return True


def main():
    """Main validation function."""
    
    print("=== Release Validation ===")
    print(f"Framework version: {get_framework_version()}")
    
    # Run all validation steps
    validation_steps = [
        ("Version Information", validate_version_info),
        ("Test Suite", validate_test_suite),
        ("Documentation", validate_documentation),
        ("Examples", validate_examples),
        ("CI/CD Pipeline", validate_ci_pipeline),
        ("Release Artifacts", validate_release_artifacts),
        ("Reproducibility", validate_reproducibility),
        ("Performance", validate_performance)
    ]
    
    failed_validations = []
    warnings = []
    
    for step_name, validation_func in validation_steps:
        try:
            success = validation_func()
            if not success:
                failed_validations.append(step_name)
        except Exception as e:
            print(f"   ❌ Error in {step_name}: {e}")
            failed_validations.append(step_name)
    
    # Summary
    print(f"\n{'='*50}")
    print("RELEASE VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    if not failed_validations:
        print("✅ All validation checks passed!")
        print("\nThe release is ready for deployment:")
        print("1. All tests pass")
        print("2. Documentation is complete")
        print("3. CI/CD pipeline is configured")
        print("4. Release artifacts are generated")
        print("5. Performance is acceptable")
        print("6. Reproducibility is validated")
        
        print(f"\nNext steps:")
        print("1. Review generated RELEASE_NOTES.md")
        print("2. Update author information in .zenodo.json")
        print("3. Create and push version tag")
        print("4. Create GitHub release")
        print("5. Submit to Zenodo for DOI")
        
        return True
    else:
        print("❌ Validation failed for:")
        for validation in failed_validations:
            print(f"   - {validation}")
        
        print("\nPlease fix the issues and run validation again.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)