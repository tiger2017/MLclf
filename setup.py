from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='MLclf',
    author="Daniel Cao",
    author_email="supercxman@gmail.com",
    description='mini-imagenet dataset transformed to fit classification task.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/tiger2017/mlclf",
    version='0.2.0',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        "Operating System :: MacOS",
    ],
    python_requires=">=3.7",
    install_requires=['numpy', 'torch']
)
