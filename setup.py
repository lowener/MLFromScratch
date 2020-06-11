import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLFromScratch", # Replace with your own username
    version="0.0.1",
    author="Mickael",
    author_email="ide.mickael@gmail.com",
    description="ML... from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lowener/MLFromScratch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL3 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)