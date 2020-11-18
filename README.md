# DiamondsInTheRough

Setup steps:
1. Verify that you have JDK >= 1.8 with `java -version`
2. Install the environment with `conda env create -f environment.yml`
3. Activate the diamonds environment with `conda activate diamonds`
4. Download the dataset with `python setup/download_data.py`. This script will download all datasets (65.2GB) to the provided directory.
5. MineRL does [not work in headless environments](https://minerl.io/docs/tutorials/index.html). If you wish to train & experiment in a headless environment, your commands should be preceded by a renderer like `xvfb-run`. Make sure this is installed properly.
