from setuptools import setup

# Read the contents of the README file into the long_description
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='goodpoints',
    version='0.0.6',
    description='Tools to generate concise high-quality summaries of a probability distribution',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/goodpoints',
    author='Raaz Dwivedi, Lester Mackey, and Abhishek Shetty',
    license='MIT',
    packages=['goodpoints'],
    install_requires=['numpy']
    )
