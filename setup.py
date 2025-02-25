import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featurewiz_polars",
    version="0.1.4",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Fast feature selection using MRMR algorithm and Polars for large datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/featurewiz_polars",
    py_modules = ["featurewiz_polars"],
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "numpy<2.0",
        "pandas>=1.2.4",
        "scipy",
        "scikit-learn>=1.2.2",
        "xgboost>=1.6",
        "polars",
        "pyarrow",
        "kneed",
    ],
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
