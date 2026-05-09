from setuptools import find_packages
from setuptools import setup

setup(
    name='card_scanner',
    version='1.0.0',
    packages=find_packages(
        include=('card_scanner', 'card_scanner.*')),
)
