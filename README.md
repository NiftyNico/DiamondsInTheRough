# DiamondsInTheRough

MineRL Environment Setup steps:
1. Verify that you have JDK >= 1.8 with `java -version`
2. Install the environment with `conda env create -f environment.yml`
3. Activate the diamonds environment with `conda activate diamonds`
4. Download the dataset with `python setup/download_data.py`. This script will download all datasets (65.2GB) to the provided directory.
5. MineRL does [not work in headless environments](https://minerl.io/docs/tutorials/index.html). If you wish to train & experiment in a headless environment, your commands should be preceded by a renderer like `xvfb-run`. Make sure this is installed properly.


Running Visualization:
1. Put vis/visualization.py in mod directory in one of the baselines
2. Run get_images() method in main() to generate potential input images
3. In main(), set image to the path of your chosen input image and set model_path to the path of the model to create visualizations for
4. In main(), run get_convs(image, model_path)

Running DQfD: 

1. Clone git repo https://github.com/marioyc/minerl2020_dqfd_submission
2. Copy project_train.py, cached_kmeans.py, and utils.py from our submission repo /instructions/dqfd into main directory
3. Run python project_train.py or xvfb-run -a python project_train.py (for run without connected monitor) with args (below) 

project_train.py --env MineRLTreechopVectorObf-v0 --outdir result --gpu 0 --noisy-net-sigma 0.5 --replay-buffer-size 300000 --replay-start-size 5000 --target-update-interval 10000 --num-step-return 10 --lr 0.0000625 --frame-stack 4 --frame-skip 4 --gamma 0.99 --batch-accumulator mean --n-pretrain-steps 10000 --n-pretrain-rounds 0 --n-experts 25 --use-noisy-net before-pretraining --only_pretrain 0 --optimizer adam --load 'load_model_directory'

Added args (compared to default): 
--n-pretrain-rounds: number of pertaining rounds (after each round, the agent gets saved)
--n-pretrain-steps: now refers to steps per pretrain round (before: total pretrain steps)
--only_pretrain: only pretraining & evaluation of pretrained agent (no actual training)

