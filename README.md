# MinimalCadQuery

CadQuery is an intuitive, easy-to-use Python module for constructing parametric 3D CAD models. Originally created by dcowden, it is now being developed under the CadQuery organization.

MinimalCadQuery is a fork of CadQuery, offering a minimal implementation with the OCP core.

Its primary objectives include:

1. Understanding CadQuery’s source code.
2. Exploring new ideas.
3. Experimenting with adaptation to other CAD kernels.

# Build and upload minimalcadquery
The following commands should be executed in a Python environment which installed twine package.
```
python -m build
python -m twine upload --repository testpypi dist/*
```

# Using minimalcadquery in a virtual environment
```
python -m venv mcpvenv
python -m pip install --upgrade pip
pip install wheel
python -m pip install --extra-index-url https://test.pypi.org/simple/ --no-build-isolation minimalcadquery==0.1.0
```

# For no-deps installation
```
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps minimalcadquery==0.1.0
```
# For local developing and testing
```
(mcq) PS D:\development\hobbies\CAD\MinimalCadQuery\dist> pip show minimalcadquery
WARNING: Package(s) not found: minimalcadquery

(mcq) PS D:\development\hobbies\CAD\MinimalCadQuery\dist> pip install minimalcadquery-0.1.12-py3-none-any.whl

(mcq) PS D:\development\hobbies\CAD\MinimalCadQuery\dist> pip show minimalcadquery

(mcq) PS D:\development\hobbies\CAD\MinimalCadQuery\dist> pip uninstall minimalcadquery
```

# Misc issues
pip says version 40.8.0 of setuptools does not satisfy requirement of setuptools>=40.8.0
```
pip install --no-index --find-links=deps --no-build-isolation psycopg[c]
```

Can't build wheel - error: invalid command 'bdist_wheel'
Install virtualenv and pip in the global directory,create your environment, activate it and install wheel. 
```
$ python3 -m pip install wheel
```

# Misc
Q：What's the difference the Point in sketch.py and the Point in hull.py？
```
Point = Union[Vector, Tuple[Real, Real]]
class Point:
```
vtk deps are brought by ocp
