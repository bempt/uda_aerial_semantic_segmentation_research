from setuptools import setup, find_packages

setup(
    name="uda_aerial_segmentation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchmetrics',
        'segmentation-models-pytorch',
        'tensorboard',
        'pandas',
        'numpy',
        'pillow',
        'albumentations'
    ],
) 