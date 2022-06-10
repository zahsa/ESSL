import os
from setuptools import setup

setup(
    name="essl",
    version="0.1",
    description="Pretext Task Learning and Optimization",
    author="Zahra Sadeghi, Noah Barrett",
    include_package_data=True,
    zip_safe=False,
    install_requires=['absl-py==1.0.0', 'aiohttp==3.8.1', 'aiosignal==1.2.0',
                      'antlr4-python3-runtime==4.8', 'async-timeout==4.0.2',
                      'attrs==21.4.0', 'cachetools==5.0.0', 'certifi==2021.10.8',
                      'charset-normalizer==2.0.12', 'deap==1.3.1', 'frozenlist==1.3.0',
                      'fsspec==2022.3.0', 'google-auth==2.6.6', 'google-auth-oauthlib==0.4.6',
                      'grpcio==1.44.0', 'hydra-core==1.1.2', 'idna==3.3', 'importlib-metadata==4.11.3',
                      'joblib==1.1.0', 'lightly==1.2.13', 'lightly-utils==0.0.2', 'Markdown==3.3.6',
                      'multidict==6.0.2', 'numpy', 'oauthlib==3.2.0', 'omegaconf==2.1.2',
                      'packaging==21.3', 'pandas', 'Pillow', 'protobuf==3.20.1',
                      'pyasn1==0.4.8', 'pyasn1-modules==0.2.8', 'pyDeprecate==0.3.2', 'pyparsing==3.0.8',
                      'python-dateutil==2.8.2', 'pytorch-lightning==1.6.1', 'pytz==2022.1', 'PyYAML==6.0',
                      'requests==2.27.1', 'requests-oauthlib==1.3.1', 'rsa==4.8', 'scikit-learn==1.0.2',
                      'scipy==1.8.0', 'six==1.16.0', 'sklearn==0.0', 'tensorboard==2.8.0',
                      'tensorboard-data-server==0.6.1', 'tensorboard-plugin-wit==1.8.1',
                      'threadpoolctl==3.1.0', 'torch', 'torchmetrics==0.8.0',
                      'torchvision', 'tqdm', 'typing_extensions',
                      'urllib3', 'Werkzeug', 'yarl', 'zipp'],
    entry_points={
        "console_scripts": ["essl_train = essl.main:main_cli"]
    },
)
