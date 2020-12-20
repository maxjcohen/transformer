"""
Setup
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="time_series_transformer",
    version="0.3.0",
    author="Max Cohen",
    author_email="max.zagouri@pm.me",
    description="Time Series Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://timeseriestransformer.readthedocs.io/",
    packages=['time_series_transformer'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "numpy==1.19.0"
    ],
    extras_require={
        'dev': [
            'pytest',
            'pylint',
            'autopep8',
            'jupyterlab',
            'matplotlib',
            'seaborn',
            'tqdm',
            'python-dotenv',
            'python-dotenv[cli]',
            'psutil',
            'lxml'
        ]
    }
)
