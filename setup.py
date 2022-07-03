from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='MLclf',
    author="Daniel Cao",
    author_email="supercxman@gmail.com",
    description='mini-imagenet and tiny-imagenet dataset transformed to fit classification task or keep the format for few-shot learning / meta-learning tasks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/tiger2017/mlclf",
    version='0.2.14',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=['numpy', 'torch', 'torchvision']
)
