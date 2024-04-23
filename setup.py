from setuptools import setup, find_packages

NAME = "I3MasterClassHarvard2024"
URL = "https://github.com/kcarloni/IceCube_MasterClass_at_Harvard2024"

print(find_packages())

setup(
    name=NAME,
    url=URL,
    packages=find_packages(where='./src/'),
    install_requires=[],
    setup_requires=["setuptools>=34.4.0"],
)