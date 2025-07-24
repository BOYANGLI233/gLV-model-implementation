from setuptools import setup, find_packages

setup(
    name='source',
    version='0.1',
    packages=find_packages(where='source'),
    package_dir={'': 'source'},
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    python_requires='>=3.6',
)