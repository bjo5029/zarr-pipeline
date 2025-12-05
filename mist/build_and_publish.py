#!/usr/bin/env python3
"""
Build and publish script for MIST package.

This script helps build and publish the MIST package to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result


def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    dirs_to_clean = ["build", "dist", "*.egg-info"]
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed {path}")
            else:
                path.unlink()
                print(f"Removed {path}")


def build_package():
    """Build the package."""
    print("Building package...")
    run_command("python -m build")


def check_package():
    """Check the package for common issues."""
    print("Checking package...")
    run_command("python -m twine check dist/*")


def upload_to_test_pypi():
    """Upload to Test PyPI."""
    print("Uploading to Test PyPI...")
    run_command("python -m twine upload --repository testpypi dist/*")


def upload_to_pypi():
    """Upload to PyPI."""
    print("Uploading to PyPI...")
    run_command("python -m twine upload dist/*")


def install_package():
    """Install the package in development mode."""
    print("Installing package in development mode...")
    run_command("pip install -e .")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python build_and_publish.py <command>")
        print("Commands:")
        print("  clean     - Clean build artifacts")
        print("  build     - Build the package")
        print("  check     - Check the package")
        print("  test      - Upload to Test PyPI")
        print("  publish   - Upload to PyPI")
        print("  install   - Install in development mode")
        print("  all       - Clean, build, check, and upload to Test PyPI")
        return

    command = sys.argv[1]

    if command == "clean":
        clean_build()
    elif command == "build":
        build_package()
    elif command == "check":
        build_package()
        check_package()
    elif command == "test":
        build_package()
        check_package()
        upload_to_test_pypi()
    elif command == "publish":
        build_package()
        check_package()
        upload_to_pypi()
    elif command == "install":
        install_package()
    elif command == "all":
        clean_build()
        build_package()
        check_package()
        upload_to_test_pypi()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
