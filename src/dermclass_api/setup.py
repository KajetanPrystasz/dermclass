import io
import os
import yaml
from pathlib import Path

from setuptools import find_packages, setup

NAME = 'dermclass_api'
DESCRIPTION = 'This package...'
EMAIL = 'kajetan.prystasz@gmail.com'
AUTHOR = 'Kajetan Prystasz'


here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'dermclass_api'
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()


def list_reqs(file_name: str= ROOT_DIR / 'environment.yml') -> list:
    """List necessary packages from requirements.txt"""
    with open(file_name) as file:
        yaml_list = yaml.load(file, Loader=yaml.FullLoader)["dependencies"]
        pip_list = yaml_list.pop(-1)["pip"]
        for package_lists in (yaml_list, pip_list):
            try:
                # Handle additional data written from conda export
                packages = [package.split("=")[:2] for package in package_lists]
                packages = ["==".join(package) for package in packages]
            except:
                packages = [package.replace("=", "==") for package in package_lists]

        return packages



setup(
    name=NAME,
    version=_version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(exclude=('tests',)),
    package_data={'dermclass_api': ['VERSION']},
    install_requires=list_reqs(),
    include_package_data=True,
)
