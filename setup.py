from setuptools import setup, find_packages

setup(
    name='source',
    version='0.1',
    package_dir={'': 'source'},
    packages=find_packages(where='source'),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    python_requires ='>=3.6',
)