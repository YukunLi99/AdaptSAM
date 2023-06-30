import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


setup(
    name="AdaptSAM",
    version="1.0",
    description="Adapt Segment Anything Model Based on Detectron2",
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires=">=3.8",
    install_requires=[
        "timm==0.6.11",  # freeze timm version for stabliity
        "opencv-python==4.6.0.66",
        "diffdist==0.1",
        "nltk>=3.6.2",
        "einops>=0.3.0",
        "wandb>=0.12.11",
        # "transformers==4.20.1",  # freeze transformers version for stabliity
        # there is BC breaking in omegaconf 2.2.1
        # see: https://github.com/omry/omegaconf/issues/939
        "omegaconf==2.1.1",
        "open-clip-torch==2.0.2",
    ],
    extras_require={
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "flake8-bugbear",
            "flake8-comprehensions",
            "click==8.0.4",
            "importlib-metadata==4.11.3",
        ],
    },
    include_package_data=True,
)