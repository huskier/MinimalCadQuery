# MinimalCadQuery

CadQuery is an intuitive, easy-to-use Python module for constructing parametric 3D CAD models. Originally created by dcowden, it is now being developed under the CadQuery organization.

MinimalCadQuery is a fork of CadQuery, offering a minimal implementation with the OCP core.

Its primary objectives include:

1. Understanding CadQuery’s source code.
2. Exploring new ideas.
3. Experimenting with adaptation to other CAD kernels.

# Build and upload minimalcadquery
python -m build
python -m twine upload --repository testpypi dist/*

# Using minimalcadquery in a virtual environment
python -m venv mcpvenv
python -m pip install --upgrade pip
pip install wheel
python -m pip install --extra-index-url https://test.pypi.org/simple/ --no-build-isolation minimalcadquery==0.0.5

# For no-deps installation
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps minimalcadquery==0.0.7

# Misc issues
pip says version 40.8.0 of setuptools does not satisfy requirement of setuptools>=40.8.0
pip install --no-index --find-links=deps --no-build-isolation psycopg[c]            【--no-build-isolation】

Can't build wheel - error: invalid command 'bdist_wheel'
Install virtualenv and pip in the global directory,create your environment, activate it and install wheel. 
$ python3 -m pip install wheel After that you do you.

# Misc
Q：What's the difference the Point in sketch.py and the Point in hull.py？
Point = Union[Vector, Tuple[Real, Real]]
class Point:

vtk deps are brought by ocp

vtk
typing-extensions
IPython

Does
pip install --index-url https://test.pypi.org/simple/  multimethod==1.9.1
work? I would assume not, since you are not allowing pip to access PyPI, only test PyPI. --extra-index-url allows pip to access both.
