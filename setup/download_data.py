# System Dependencies
import os 

# External Dependencies
import minerl

path = input("Path to download dataset (default 'git_root/raw_data/'):")
path = path.strip()

if not path:
  git_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
  path     = os.path.join(git_path, 'raw_data')

minerl.data.download(directory=path)