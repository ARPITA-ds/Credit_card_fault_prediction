from setuptools import setup
from typing import List


PROJECT_NAME="Credit card fault prediction"
VERSION = "0.0.1"
AUTHOR = "Arpita Maity"
PACKAGES = ["src\CreditCard"]
REQUIREMENT_FILE_NAME = "requirements.txt"

def get_requirements_list()->List[str]:
    """
    This function is going to return list of requirements
    mention in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        return requirement_file.readlines()

setup(
name=PROJECT_NAME,
version=VERSION,
author=AUTHOR,
packages=PACKAGES,
install_requires= get_requirements_list()
)

