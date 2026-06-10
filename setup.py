from setuptools import setup, find_packages

setup(
    name="gymnasium_stag_hunt",
    version="0.0.1",
    author="Giorgio Franceschelli - fork from David Nesterov-Rappoport",
    author_email="giorgio.franceschelli@unibo.it",
    description="Markov stag hunt environment for gymnasium",
    long_description="This package is based on gymnasium and a fork from the original OpenAI gym-based stag-hunt "
    "environment.",
    long_description_content_type="text/markdown",
    url="https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt",
    packages=find_packages(),
    include_package_data=True,
    package_data={'gymnasium_stag_hunt': ['assets/*', 'assets/**/*']},
    install_requires=["gymnasium", "pygame", "opencv-python", "pettingzoo"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
