from setuptools import setup, find_packages

setup(
    name="pyEchoStateNetwork",
    version="1.0.0",
    description="A simple python library for designing and testing Echo State Networks.",
    author="Dafydd Heyburn",
    packages=find_packages(),
    install_requires=[
        "numpy >=2.2.1",
        "pandas >= 2.2.3",
        "scipy >= 1.15.0",
        "matplotlib >= 3.10.0",
        "pytest >= 8.3.4",
    ],
    python_requires=">=3.10"
)
