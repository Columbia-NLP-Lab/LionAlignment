from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="LionAlignment",
    version="0.0.1",
    description="Columbia Lion NLP Alignment Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['lionalign'],
    install_requires=required_packages,
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)