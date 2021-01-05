import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
<<<<<<< HEAD
    name="tst",
    version="0.3.0",
    author="Max Cohen",
    author_email="max.zagouri@pm.me",
=======
    name="time_series_transformer",
    version="1.0.0",
    author="Daniel Kaminski de Souza",
    author_email="daniel@kryptonunite.com",
>>>>>>> 27a9c3b58c74a1217403b8e008d275bd0e8ac3c5
    description="Time Series Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://timeseriestransformer.readthedocs.io/",
    packages=['tst'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "numpy==1.19.3", # not actually a dependency anylonger, just here because latest numpy, a torch dependency, at the time of writting is broken in Windows.
    ],
    extras_require={
        'dev': [
            'pylint',
            'autopep8',
            'bumpversion',
            'twine',
        ],
        'test': [
            'pytest>=4.6',
            'pytest-cov',
            'flights-time-series-dataset',
            'time-series-predictor',
        ],
        'docs': [
            'Sphinx>3'
            'sphinx-autodoc-typehints'
            'ipython'
            'nbsphinx'
            'recommonmark'
            'sphinx_rtd_theme'
        ]
    }
)
