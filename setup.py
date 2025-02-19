from setuptools import setup, find_packages

setup(
    name="superhuge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "torch",
        "lightning",
        "pydantic",
        "numpy",
        "pyyaml",
        "einops",
        # Add other dependencies here
    ],
    entry_points={
        "console_scripts": [
            # Define command-line scripts here
        ],
    },
    author="Yuanming Zhang",
    author_email="yuanming.zhang@smail.nju.edu.cn",
    description="SuperHugeAAD: extending Deep Learning-based Auditory Attention Decoding to more than one singel datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SeanZhang99/SuperHugeAAD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
