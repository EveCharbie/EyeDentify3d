import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eyedentify3d",
    version="1.0.7",
    author="Eve Charbonneau",
    author_email="eve.charbie@gmail.com",
    description="Identify gaze behaviors from 3D eye-trakcing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EveCharbie/EyeDentify3D",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# Publish on pip manually
# 1) change version in version.py, setup.py
# 2) Remove old wheel build from dist folder
# 3) python setup.py sdist bdist_wheel
# 4) python -m twine upload dist/*

# Publishing on conda-forge
# 0) conda install conda-build conda-forge-pinning
# 1) publish on pip first
# 2) download the latest eyedentify3d-{version}.tar.gz from pip
# 3) run openssl sha256 eyedentify3d-{version}.tar.gz to get the sha256 hash
# 4) paste the hash in the recipe/meta.yaml file
# 5) ?
