from setuptools import setup
from setuptools import find_packages

requirements = [
    'numpy',
    'matplotlib',
    'urllib3',
    'tqdm',
    'pillow',
    'scipy',
    'opencv-python'
    # 'torch==1.2.0',
    # 'torchvision==0.4.0'
]

setup(
    name='apfv21',
    description='Code for AAAI2022 submission APFV.',
    version='1.0',
    url="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)
