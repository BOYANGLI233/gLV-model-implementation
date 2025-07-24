from setuptools import setup, find_packages

setup(
    name="microbInfer",
    version="0.1.0",
    packages=find_packages(),
    description="A package for microbial interactions inference",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
    ],
    python_requires=">=3.10",
)