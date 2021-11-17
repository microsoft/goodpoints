#!/bin/bash -x

rm -rf build
rm -rf dist
rm -rf *egg-info


python setup.py sdist bdist_wheel

# upload commands
python -m twine upload dist/*

