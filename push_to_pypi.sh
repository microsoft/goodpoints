#!/bin/bash -x

rm -rf build
rm -rf dist
rm -rf *egg-info

python setup.py sdist bdist_wheel

# upload commands
python -m twine upload dist/*
#python -m twine upload --repository testpypi dist/*

# install from test pypi
#python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-binary :all: goodpoints==0.2.2
