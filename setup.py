import os
from setuptools import setup

setup(
    name="ESSL",
    version="0.1",
    description="Pretext Task Learning and Optimization",
    author="Zahra Sadeghi, Noah Barrett",
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "pillow",
        "deap"
    ],
    entry_points={
        "console_scripts": [
        ]
    },
)