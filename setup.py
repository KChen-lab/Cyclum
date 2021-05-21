import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # project details
    name="cyclum-shaoheng",
    version="0.2a1",
    description="Identify circular trajectories in scRNA-seq data using an autoencoder with sinusoidal activations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KChen-lab/Cyclum",
    license='MIT',
    
    # author details
    author="Shaoheng Liang",
    author_email="sliang3@mdanderson.org",

    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6, <3.8',
    install_requires=[
        'keras', 'tensorflow==2.5.0', 'numpy', 'pandas', 'scikit-learn', 'h5py', 'jupyter', 'matplotlib'] 
)
