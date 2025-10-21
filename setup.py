from setuptools import setup, find_packages

setup(
    # Package name (will be used for pip install)
    name="overcooked-marl-env",
    
    # Initial version of your environment
    version="0.0.1",
    
    # Basic information
    description="A Gymnasium/PettingZoo environment for MARL based on Overcooked.",
    author="thaisfds",
    
    # Automatically finds your code folder ('overcooked_marl_env')
    packages=find_packages(),
    
    # Dependencies required to run the environment
    install_requires=[
        "gymnasium>=0.29.1", # The most recent version of Gymnasium
        "numpy",             # For handling arrays and observations
        "pettingzoo>=1.24.0",# Essential for the ParallelEnv base class
    ],
    
    # Minimum Python version
    python_requires=">=3.10", # Recommended to use Python 3.10+
)