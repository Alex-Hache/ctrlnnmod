from setuptools import setup, find_packages


TEST_REQUIRES = ["pytest"]

DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
]

classifiers = [
    "Development Status :: 1 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
]

# Get the long description from the README file
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()


VERSION = "0.2.6"

setup(
    name="ctrlnmod",
    version=VERSION,
    description="Control oriented neural state space models and tools for sdp constrained learning algorithms.",
    author="Alexandre Hache",
    author_email="alexandre.hache@outlook.fr",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Alex-Hache/ctrlnnmod",
    classifiers=classifiers,
    keywords=["Neural networks", "Linear Matrix Inequalities", "Pytorch"],
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["torch>=1.9", "numpy", "cvxpy[MOSEK]<=1.5", "typeguard", "alive_progress", "matplotlib"],
    extras_require={"dev": DEV_REQUIRES, "test": TEST_REQUIRES},
)
