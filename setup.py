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
                      'antlr4-python3-runtime==4.8', 'asttokens==2.0.5', 'async-timeout==4.0.2',
                      'attrs==21.4.0', 'backcall==0.2.0', 'cachetools==5.0.0', 'certifi==2021.10.8',
                      'charset-normalizer==2.0.12', 'click==8.1.3', 'cycler==0.11.0', 'deap==1.3.1',
                      'debugpy==1.6.0', 'decorator==5.1.1', 'entrypoints==0.4', 'executing==0.8.3',
                      'fonttools==4.33.3', 'frozenlist==1.3.0', 'fsspec==2022.3.0', 'google-auth==2.6.6',
                      'google-auth-oauthlib==0.4.6', 'grpcio==1.44.0', 'hydra-core==1.1.2', 'idna==3.3',
                      'importlib-metadata==4.11.3', 'ipykernel==6.13.0', 'ipython', 'jedi==0.18.1',
                      'Jinja2==3.1.2', 'joblib==1.1.0', 'jupyter-client==7.3.1', 'jupyter-core==4.10.0',
                      'kiwisolver==1.4.3', 'lightly==1.2.23', 'lightly-utils==0.0.2', 'Markdown==3.3.6',
                      'MarkupSafe==2.1.1', 'matplotlib==3.5.2', 'matplotlib-inline==0.1.3', 'multidict==6.0.2',
                      'mycolorpy==1.5.1', 'nest-asyncio==1.5.5', 'numpy', 'oauthlib==3.2.0', 'omegaconf==2.1.2',
                      'packaging==21.3', 'pandas', 'parso==0.8.3', 'pexpect==4.8.0', 'pickleshare==0.7.5',
                      'Pillow==9.2.0', 'prompt-toolkit==3.0.29', 'protobuf==3.20.1', 'psutil==5.9.1', 'ptyprocess==0.7.0',
                      'pure-eval==0.2.2', 'pyasn1==0.4.8', 'pyasn1-modules==0.2.8', 'pyDeprecate==0.3.2', 'Pygments==2.12.0',
                      'pyparsing==3.0.8', 'python-dateutil==2.8.2', 'pytorch-lightning==1.6.1', 'pytz==2022.1', 'PyYAML==6.0',
                      'pyzmq==23.0.0', 'requests==2.27.1', 'requests-oauthlib==1.3.1', 'rsa==4.8', 'scikit-learn==1.0.2',
                      'scipy', 'seaborn==0.11.2', 'six==1.16.0', 'sklearn==0.0', 'stack-data==0.2.0', 'tensorboard==2.8.0',
                      'tensorboard-data-server==0.6.1', 'tensorboard-plugin-wit==1.8.1', 'tensorboardX==2.5.1', 'threadpoolctl==3.1.0',
                      'torch==1.12.0', 'torchmetrics==0.8.0', 'torchvision==0.13.0', 'tornado==6.1', 'tqdm==4.64.0', 'traitlets==5.2.1.post0',
                      'typing_extensions==4.3.0', 'urllib3==1.26.9', 'wcwidth==0.2.5', 'Werkzeug==2.1.2', 'yarl==1.7.2', 'zipp==3.8.0'],
    entry_points={
        "console_scripts": ["essl_GA = essl.main:GA_cli",
                            "essl_GA_bootstrap = essl.main:GA_bootstrap_cli",
                            "essl_GA_mo = essl.main:GA_mo_cli",
                            "essl_GA_mo_bootstrap = essl.main:GA_mo_bootstrap_cli",
                            "essl_ll_random_plane = essl.utils:ll_random_plane_cli",
                            "essl_ll_linear_i = essl.utils:ll_linear_interpolation_cli"
                    ]
                        },
                    )
