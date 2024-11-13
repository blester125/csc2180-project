"""Install package for easy tool access."""

import ast
import os

from setuptools import find_packages, setup


def get_version(file_name: str, version_variable: str = "__version__") -> str:
    """Find the version by walking the AST to avoid duplication.

    Parameters
    ----------
    file_name : str
        The file we are parsing to get the version string from.
    version_variable : str
        The variable name that holds the version string.

    Raises
    ------
    ValueError
        If there was no assignment to version_variable in file_name.

    Returns
    -------
    version_string : str
        The version string parsed from file_name_name.
    """
    with open(file_name) as f:
        tree = ast.parse(f.read())
        # Look at all assignment nodes that happen in the ast. If the variable
        # name matches the given parameter, grab the value (which will be
        # the version string we are looking for).
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if node.targets[0].id == version_variable:
                    return node.value.s
    raise ValueError(
        f"Could not find an assignment to {version_variable} " f"within '{file_name}'"
    )


with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="z3ml",
    version=get_version("z3ml/__init__.py"),
    description="ML model fitting using z3 (SAT/SMT).",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Brian Lester, Jay Je",
    author_email="blester125@gmail.com",
    url="https://github.com/blester125/csc2180-project",
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="SAT SMT z3 machine-learning",
    license="MIT",
    install_requires=[
        "z3-solver",
        "numpy",
        "matplotlib",
    ],
    extras_require={},
    entry_points={},
)
