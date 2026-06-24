"""Compatibility shim for older pip editable installs."""

from setuptools import find_packages, setup


setup(
    name="finprobts-bench",
    version="0.1.0",
    description="Professional benchmark library for probabilistic financial time-series forecasting models.",
    packages=find_packages(include=["finprobts", "finprobts.*"]),
    python_requires=">=3.9",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest"],
        "metrics": ["properscoring", "scoringrules"],
        "parquet": ["pyarrow"],
        "torch": ["torch"],
        "deep": ["torch", "properscoring", "scoringrules"],
    },
    entry_points={
        "console_scripts": [
            "finprobts=finprobts.cli:main",
        ]
    },
)
