from setuptools import setup, find_packages
import os
import sys

def get_requirements():
    """Get requirements based on Python version"""

    req_file = "requirements.txt"

    
    print(f"Using requirements from: {req_file}")
    
    if os.path.exists(req_file):
        with open(req_file) as f:
            return f.read().splitlines()
    else:
        print(f"Warning: {req_file} not found. Returning empty requirements list.")
        return []

setup(
    name="micaflow",
    version="0.2.0",
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
    python_requires=">=3.10",
)