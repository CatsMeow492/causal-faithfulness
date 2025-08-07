#!/usr/bin/env python3
"""
Release Preparation Script

This script prepares the repository for release by:
1. Validating that all tests pass
2. Generating release notes
3. Creating version tags
4. Preparing documentation for archival
5. Generating metadata for Zenodo DOI
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from version_info import get_environment_info, get_framework_version


def run_command(command, cwd=None, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_git_status():
    """Check git repository status."""
    print("1. Checking git repository status...")
    
    # Check if we're in a git repository
    success, _, _ = run_command("git rev-parse --git-dir")
    if not success:
        print("   ❌ Not in a git repository")
        return False
    
    # Check for uncommitted changes
    success, output, _ = run_command("git status --porcelain")
    if not success:
        print("   ❌ Failed to check git status")
        return False
    
    if output.strip():
        print("   ⚠ Warning: Uncommitted changes detected:")
        for line in output.strip().split('\n'):
            print(f"     {line}")
        
        response = input("   Continue with uncommitted changes? (y/N): ")
        if response.lower() != 'y':
            return False
    else:
        print("   ✓ Working directory is clean")
    
    # Get current branch
    success, branch, _ = run_command("git rev-parse --abbrev-ref HEAD")
    if success:
        print(f"   Current branch: {branch.strip()}")
    
    # Get latest commit
    success, commit, _ = run_command("git rev-parse HEAD")
    if success:
        print(f"   Latest commit: {commit.strip()[:8]}")
    
    return True


def run_tests():
    """Run the test suite to ensure everything is working."""
    print("\n2. Running test suite...")
    
    # Run unit tests
    print("   Running unit tests...")
    success, output, error = run_command("python -m pytest tests/unit/ -v --tb=short")
    if not success:
        print("   ❌ Unit tests failed:")
        print(f"   {error}")
        return False
    print("   ✓ Unit tests passed")
    
    # Run integration tests
    print("   Running integration tests...")
    success, output, error = run_command("python -m pytest tests/integration/ -v --tb=short")
    if not success:
        print("   ❌ Integration tests failed:")
        print(f"   {error}")
        return False
    print("   ✓ Integration tests passed")
    
    # Run toy model test
    print("   Running toy model test...")
    success, output, error = run_command("python scripts/ci_toy_model_test.py")
    if not success:
        print("   ❌ Toy model test failed:")
        print(f"   {error}")
        return False
    print("   ✓ Toy model test passed")
    
    return True


def generate_release_notes():
    """Generate release notes based on git history and project status."""
    print("\n3. Generating release notes...")
    
    version = get_framework_version()
    
    # Get git log since last tag (or all commits if no tags)
    success, tags_output, _ = run_command("git tag --sort=-version:refname")
    
    if success and tags_output.strip():
        last_tag = tags_output.strip().split('\n')[0]
        log_range = f"{last_tag}..HEAD"
        print(f"   Generating changes since tag: {last_tag}")
    else:
        log_range = "HEAD"
        print("   No previous tags found, generating full history")
    
    # Get commit history
    success, commits, _ = run_command(
        f"git log {log_range} --pretty=format:'- %s (%h)' --no-merges"
    )
    
    if not success:
        commits = "- Initial release"
    
    # Get contributor information
    success, contributors, _ = run_command(
        f"git log {log_range} --pretty=format:'%an' --no-merges | sort | uniq"
    )
    
    if not success:
        contributors = "Unknown"
    
    # Generate release notes
    release_notes = f"""# Release Notes - Version {version}

**Release Date:** {datetime.now().strftime('%Y-%m-%d')}

## Overview

This release provides a complete implementation of the causal-faithfulness metric for evaluating post-hoc explanations across different model architectures and data modalities.

## Key Features

- **Model-agnostic faithfulness metric**: Quantifies explanation quality through causal intervention semantics
- **Multi-modal support**: Works with text, tabular, and image data
- **Theoretical foundation**: Satisfies key axioms (Causal Influence, Sufficiency, Monotonicity, Normalization)
- **Statistical rigor**: Built-in confidence intervals and significance testing
- **Hardware optimization**: Efficient computation on Mac M-series, CUDA, and CPU
- **Comprehensive testing**: Unit, integration, and validation test suites
- **Reproducibility framework**: Ensures consistent results across runs and platforms

## Supported Explanation Methods

- SHAP (KernelSHAP, TreeSHAP, DeepSHAP)
- Integrated Gradients
- LIME
- Random baseline (for sanity checking)

## Changes in This Release

{commits}

## Contributors

{contributors.replace(chr(10), ', ') if success else 'Project Team'}

## Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Quick Start

```python
from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
from src.explainers import SHAPWrapper

# Configure metric
config = FaithfulnessConfig(n_samples=1000, random_seed=42)
metric = FaithfulnessMetric(config)

# Evaluate explainer
result = metric.compute_faithfulness_score(model, explainer, data)
print(f"Faithfulness score: {{result.f_score:.4f}}")
```

## Documentation

- API Documentation: `docs/API.md`
- Tutorial: `docs/TUTORIAL.md`
- Configuration Guide: `docs/CONFIGURATION.md`
- Reproducibility Guide: `docs/REPRODUCIBILITY.md`

## Testing

Run the test suite:
```bash
pytest tests/ -v
python scripts/ci_toy_model_test.py
```

## Citation

If you use this work in your research, please cite:

```bibtex
@software{{causal_faithfulness_metric_{version.replace('.', '_')},
  title={{A Causal-Faithfulness Score for Post-hoc Explanations Across Model Architectures}},
  author={{[Authors]}},
  version={{{version}}},
  year={{{datetime.now().year}}},
  url={{[Repository URL]}},
  doi={{[Zenodo DOI]}}
}}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the examples in the `examples/` directory
"""

    # Save release notes
    with open("RELEASE_NOTES.md", "w") as f:
        f.write(release_notes)
    
    print("   ✓ Release notes generated: RELEASE_NOTES.md")
    return True


def create_zenodo_metadata():
    """Create metadata file for Zenodo DOI."""
    print("\n4. Creating Zenodo metadata...")
    
    version = get_framework_version()
    
    # Get git information
    success, commit_hash, _ = run_command("git rev-parse HEAD")
    commit_hash = commit_hash.strip() if success else "unknown"
    
    success, repo_url, _ = run_command("git remote get-url origin")
    repo_url = repo_url.strip() if success else "unknown"
    
    zenodo_metadata = {
        "title": "A Causal-Faithfulness Score for Post-hoc Explanations Across Model Architectures",
        "description": "A model-agnostic metric that quantifies how faithfully feature-based explanations reflect a model's true decision logic through principled causal intervention semantics.",
        "creators": [
            {
                "name": "[Author Name]",
                "affiliation": "[Institution]",
                "orcid": "[ORCID ID]"
            }
        ],
        "keywords": [
            "explainable AI",
            "XAI",
            "explanation faithfulness",
            "causal intervention",
            "model interpretability",
            "post-hoc explanations",
            "SHAP",
            "LIME",
            "Integrated Gradients"
        ],
        "license": "MIT",
        "upload_type": "software",
        "version": version,
        "publication_date": datetime.now().strftime('%Y-%m-%d'),
        "related_identifiers": [
            {
                "identifier": repo_url,
                "relation": "isSupplementTo",
                "resource_type": "software"
            }
        ],
        "contributors": [
            {
                "name": "[Contributor Name]",
                "type": "Other"
            }
        ],
        "references": [
            "Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. NAACL-HLT.",
            "Adebayo, J., et al. (2018). Sanity checks for saliency maps. NeurIPS.",
            "Hooker, S., et al. (2019). A benchmark for interpretability methods in deep neural networks. NeurIPS."
        ],
        "notes": f"Software implementation accompanying the research paper. Git commit: {commit_hash[:8]}",
        "language": "eng",
        "access_right": "open"
    }
    
    # Save Zenodo metadata
    with open(".zenodo.json", "w") as f:
        json.dump(zenodo_metadata, f, indent=2)
    
    print("   ✓ Zenodo metadata created: .zenodo.json")
    print("   Note: Please update author information and ORCID IDs before submission")
    return True


def create_version_tag():
    """Create a git tag for the release version."""
    print("\n5. Creating version tag...")
    
    version = get_framework_version()
    tag_name = f"v{version}"
    
    # Check if tag already exists
    success, existing_tags, _ = run_command("git tag -l")
    if success and tag_name in existing_tags:
        print(f"   ⚠ Tag {tag_name} already exists")
        response = input(f"   Overwrite existing tag? (y/N): ")
        if response.lower() == 'y':
            success, _, error = run_command(f"git tag -d {tag_name}")
            if not success:
                print(f"   ❌ Failed to delete existing tag: {error}")
                return False
        else:
            print("   Skipping tag creation")
            return True
    
    # Create annotated tag
    tag_message = f"Release version {version}"
    success, _, error = run_command(f'git tag -a {tag_name} -m "{tag_message}"')
    
    if not success:
        print(f"   ❌ Failed to create tag: {error}")
        return False
    
    print(f"   ✓ Created tag: {tag_name}")
    print(f"   To push tag to remote: git push origin {tag_name}")
    return True


def generate_archive_package():
    """Generate a complete archive package for distribution."""
    print("\n6. Generating archive package...")
    
    version = get_framework_version()
    archive_name = f"causal-faithfulness-metric-v{version}"
    
    # Create archive directory
    archive_dir = Path(archive_name)
    if archive_dir.exists():
        import shutil
        shutil.rmtree(archive_dir)
    
    archive_dir.mkdir()
    
    # Copy essential files
    essential_files = [
        "src/",
        "tests/",
        "examples/",
        "docs/",
        "scripts/",
        "requirements.txt",
        "LICENSE",
        "README.md",
        "RELEASE_NOTES.md",
        ".zenodo.json"
    ]
    
    import shutil
    
    for item in essential_files:
        src_path = Path(item)
        if src_path.exists():
            dst_path = archive_dir / item
            
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
    
    # Generate environment info
    env_info = get_environment_info()
    with open(archive_dir / "environment_info.json", "w") as f:
        json.dump(env_info, f, indent=2, default=str)
    
    # Create archive
    shutil.make_archive(archive_name, 'zip', '.', archive_name)
    
    # Clean up directory
    shutil.rmtree(archive_dir)
    
    print(f"   ✓ Archive created: {archive_name}.zip")
    return True


def validate_release_readiness():
    """Validate that the release is ready."""
    print("\n7. Validating release readiness...")
    
    checks = []
    
    # Check that all required files exist
    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "src/version_info.py",
        "docs/API.md",
        "docs/TUTORIAL.md",
        "examples/basic_usage.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            checks.append((f"Required file {file_path}", True))
        else:
            checks.append((f"Required file {file_path}", False))
    
    # Check that version is properly set
    version = get_framework_version()
    version_valid = len(version.split('.')) >= 3 and version != "0.0.0"
    checks.append(("Version properly set", version_valid))
    
    # Check that tests pass (already done, but verify)
    test_files_exist = (
        Path("tests/unit").exists() and
        Path("tests/integration").exists() and
        Path("scripts/ci_toy_model_test.py").exists()
    )
    checks.append(("Test files exist", test_files_exist))
    
    # Check documentation
    docs_complete = all(Path(f"docs/{doc}.md").exists() for doc in ["API", "TUTORIAL", "CONFIGURATION", "REPRODUCIBILITY"])
    checks.append(("Documentation complete", docs_complete))
    
    # Print validation results
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("   ✅ Release validation passed!")
    else:
        print("   ❌ Release validation failed - fix issues before releasing")
    
    return all_passed


def main():
    """Main release preparation function."""
    
    print("=== Release Preparation Script ===")
    print(f"Framework version: {get_framework_version()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all preparation steps
    steps = [
        ("Git status check", check_git_status),
        ("Test suite", run_tests),
        ("Release notes", generate_release_notes),
        ("Zenodo metadata", create_zenodo_metadata),
        ("Version tag", create_version_tag),
        ("Archive package", generate_archive_package),
        ("Release validation", validate_release_readiness)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        try:
            success = step_function()
            if not success:
                failed_steps.append(step_name)
        except Exception as e:
            print(f"   ❌ Error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print(f"\n{'='*50}")
    print("RELEASE PREPARATION SUMMARY")
    print(f"{'='*50}")
    
    if not failed_steps:
        print("✅ All preparation steps completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated RELEASE_NOTES.md")
        print("2. Update author information in .zenodo.json")
        print("3. Push the version tag: git push origin v" + get_framework_version())
        print("4. Create a GitHub release using the release notes")
        print("5. Upload to Zenodo for DOI generation")
        print("6. Update README.md with the DOI badge")
        
        return True
    else:
        print("❌ Some preparation steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease fix the issues and run the script again.")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)