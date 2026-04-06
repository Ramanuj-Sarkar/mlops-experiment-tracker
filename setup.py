from setuptools import setup, find_packages

setup(
    name="mlops-experiment-tracker",
    version="0.1.0",
    description="A lightweight, dependency-free experiment tracker for ML projects",
    author="Ramanuj Sarkar",
    packages=find_packages(),
    python_requires=">=3.8",
    extras_require={
        "reports": ["pandas>=1.3.0"],
        "examples": ["scikit-learn>=1.0.0"],
        "dev": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "tracker=tracker.__main__:main",
        ],
    },
)
