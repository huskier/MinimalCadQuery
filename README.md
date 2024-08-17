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