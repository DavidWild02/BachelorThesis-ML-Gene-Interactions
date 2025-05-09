import sys
import os
import shutil
import subprocess


# Check if this gets executed in google colab. 
# If so, then we need to install pip packages and clone the repo and data
try:
    import google.colab
except ImportError:
    IN_COLAB = False
else:
    IN_COLAB = True


if not IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
else:
    # the version used in the requirements.txt are not all compatible with colab
    # So install the needed packages manually here
    packages = [
        "scanpy",
        "torch",
        "matplotlib",
        "pandas",
        "numpy",
        "seaborn",
        "scipy",
        "statsmodels",
        "PyDrive",
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    subprocess.check_call(["git", "clone", "https://github.com/DavidWild02/BachelorThesis-ML-Gene-Interactions.git"])
    os.chdir("./BachelorThesis-ML-Gene-Interactions")

    from google.colab import drive
    drive.mount('/content/drive')
    shutil.copytree("/content/drive/MyDrive/DavidWildBachelorthesis/data", ".")


