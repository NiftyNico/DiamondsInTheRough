# System Dependencies
import os 

# External Dependencies
import minerl


path = input("Path to download dataset (default 'git_root/raw_data/'):")
path = path.strip()

if not path:
  git_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
  path     = os.path.join(git_path, 'raw_data')
  if not os.path.exists(path):
    os.makedirs(path)


print(f"-----\nNOT FINISHED\nPlease add 'MINERL_DATA_ROOT' to your bash profile\nSomething like `echo 'export MINERL_DATA_ROOT={path}' >>~/.bash_profile`\nRestart the shell when done\n-----")
minerl.data.download(directory=path)