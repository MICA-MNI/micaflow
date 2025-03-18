from setuptools import setup, find_packages
import os
import sys

def get_requirements():
    """Get requirements based on Python version"""
    python_version = sys.version_info
    
    if python_version.major == 3 and python_version.minor == 9:
        req_file = "requirements_39.txt"
    elif python_version.major == 3 and python_version.minor == 10:
        req_file = "requirements_310.txt"
    elif python_version.major == 3 and python_version.minor == 11:
        req_file = "requirements_311.txt"
    else:
        # Default to 3.9 requirements if version not specifically supported
        print(f"Warning: Python {python_version.major}.{python_version.minor} not explicitly supported. Using Python 3.9 requirements.")
        req_file = "requirements_39.txt"
    
    print(f"Using requirements from: {req_file}")
    
    if os.path.exists(req_file):
        with open(req_file) as f:
            return f.read().splitlines()
    else:
        print(f"Warning: {req_file} not found. Returning empty requirements list.")
        return []

setup(
    name="micaflow",
    version="0.1.0",
    description="Clinical and research MRI processing workflow",
    author="MICA Lab",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "micaflow": [
            "resources/Snakefile",
            "resources/atlas/*",
        ]
    },
    entry_points={
        "console_scripts": [
            "micaflow=micaflow.cli:main",
        ],
    },
    install_requires=get_requirements(),
    python_requires=">=3.9",
)