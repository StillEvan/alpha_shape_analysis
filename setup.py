import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="alpha_shape_analysis-StillEvan", # Replace with your own username
    version="0.0.1",
    author="Evan Still",
    author_email="evanstill@berkeley.edu",
    description="A package for determining volume, surface area, and composition of 3-d clusters for Atom Probe Tomograhpic data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StillEvan/alpha_shape_analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)