from setuptools import setup, find_packages

setup(
    name="multielo-football",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "optuna>=3.0.0",
        "matplotlib>=3.4.0"
    ],
    author="César Rennó-Costa, László Csató",
    author_email="cesar@imd.ufrn.br",
    description="Multi-Dimensional Elo Ratings and Poisson Prediction Hierarchy for International Football",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/rennocosta/matchdataset",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
