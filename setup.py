# coding=utf-8

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

reqs = [
    "torch>=2.0.1",
    "torchvision>=0.15.2", 
    "scikit-learn>=1.3.0,<=1.4.0",
    "pandas>=2.0.0,<=2.1.4",
    "numpy>=1.24.0,<=1.26.2",
    "scipy>=1.10.0,<=1.11.4",
    "attrs>=23.0.0",
    "flatten_dict>=0.4.2",
    "tensorboardX>=2.6.0",
    "gcsfs>=2023.6.0",
    "google-auth>=2.23.0",
    "fsspec>=2023.6.0"
]

setup(
    name="allRank",
    version="1.4.3",
    description="allRank is a framework for training learning-to-rank neural models",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache 2",
    url="https://github.com/allegro/allRank",
    install_requires=reqs,
    author_email="allrank@allegro.pl",
    packages=find_packages(exclude=["tests"]),
    package_data={"allrank": ["config.json"]},
    entry_points={"console_scripts": ['allRank = allrank.main:run']},
    zip_safe=False,
)
