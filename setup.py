from setuptools import setup, find_packages
setup(
    name = 'machine_learning',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'sphinx',
        'sphinx_rtd_theme',
        'ghp-import'
    ],
)