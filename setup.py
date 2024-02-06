from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="ssl_neuron",
    version="1.1",
    description="SSL-Neuron contains the code to the paper 'Self-supervised Representation Learning of Neuronal Morphologies'",
    author="Marissa Weis",
    author_email="marissa.weis@bethgelab.org",
    url="https://eckerlab.org/code/weis2021b/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
)
