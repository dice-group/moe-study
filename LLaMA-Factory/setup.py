from setuptools import setup, find_packages

setup(
    name="llamafactory",
    version="0.9.2+moe_study",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "llamafactory-cli = llamafactory.cli:main",
        ],
    },
)
