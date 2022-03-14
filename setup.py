from setuptools import setup

setup(
    name='castep_elnes_parser',
    version="1.0.1",
    description="A simple python script for parsing CASTEP output files especially for ELNES calculations.",
    long_description="A simple python script for parsing [CASTEP](http://www.castep.org/) output files especially for ELNES calculations. Contains some functions for calculating excitation energies and forming spectra",
    url='https://github.com/nmdl-mizo/castep_elnes_parser',
    author='kiyou, nmdl-mizo',
    author_email='',
    license='MIT',
    classifiers=[
        # https://pypi.python.org/pypi?:action=list_classifiers
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
    ],
    keywords='tools',
    install_requires=["numpy"],
    py_modules=["castep_elnes_parser"],
)
