from setuptools import setup, find_packages

setup(
    name='CReSS',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch', 
        'torchvision',
        'transformers',
        'tqdm',
    ],
)
