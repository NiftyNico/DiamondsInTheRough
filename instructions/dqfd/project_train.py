# THis is a test

import json
import select
import time
import logging
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from pip._internal.operations import freeze
import aicrowd_helper
from utility.parser import Parser
import utils

import coloredlogs
import argparse
from collections import deque
from inspect import getsourcefile

import gym
import minerl  # noqa: register MineRL envs as Gym envs.
import numpy as np
import pickle
from torch import optim
import tqdm

import pfrl
from pfrl import experiments, explorers
from pfrl.experiments.evaluator import Evaluator
from pfrl.wrappers import ContinuingTimeLimit, RandomizeAction
from pfrl.wrappers.atari_wrappers import LazyFrames, ScaledFloatFrame

from dqfd import DQfD, PrioritizedDemoReplayBuffer

from q_functions import DuelingDQN
from env_wrappers import (
    ClusteredActionWrapper,
    MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper,
    PoVWithCompassAngleWrapper, FullObservationSpaceWrapper)
from cached_kmeans import cached_kmeans
# import coloredlogs
# coloredlogs.install(logging.INFO)


# logger = logging.getLogger(__name__)

def get_kmeans_clustering(args, logger):
    logger.info("Starting KMeans clustering")
    dat = minerl.data.make(args.env)
    act_vectors = []
    for _, act, _, _,_ in tqdm.tqdm(dat.batch_iter(1, 32, 2, preload_buffer_size=20)):
        act_vectors.append(act['vector'])
        if len(act_vectors) > 1000:
            break
    acts = np.concatenate(act_vectors).reshape(-1, 64)
    kmeans_acts = acts[:100000]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(kmeans_acts)
    logger.info("Completed KMeans clustering")
    return kmeans


def fill_buffer(replay_buffer, args, kmeans, logger):
    logger.info("Fill the buffer with expert data")
    dat = minerl.data.make(args.env)
    trajectories = dat.get_trajectory_names()
    logger.info("%s trajectories", len(trajectories))

    n_experts = min(args.n_experts, len(trajectories))

    for trajectory_id in range(n_experts):
        trajectory = trajectories[trajectory_id]
        first = True
        frames = []
        actions = []
        rewards = []

        for (state, a, r, next_state, done, meta) in dat.load_data(trajectory, include_metadata=True):
            state = state['pov']
            state = np.moveaxis(state, -1, 0)
            next_state = next_state['pov']
            next_state = np.moveaxis(next_state, -1, 0)

            if first:
                first = False
                logger.info("name: %s, success: %s, steps: %s, reward: %s",
                            meta["stream_name"], meta["success"],
                            meta["duration_steps"], meta["total_reward"])
                frames.append(state)
            frames.append(next_state)
            actions.append(kmeans.predict(a['vector'].reshape(1, -1))[0])
            rewards.append(r)

        obs_q = deque([], maxlen=args.frame_stack)
        for i in range(args.frame_stack):
            obs_q.append(frames[0])

        for i in range(0, len(frames) - 1, args.frame_skip):
            if i + args.frame_skip >= len(frames) - 1:
                done = True
            else:
                done = False
            state = LazyFrames(list(obs_q), stack_axis=0)
            next_frame = frames[min(i + args.frame_skip, len(frames) - 1)]
            obs_q.append(next_frame)
            next_state = LazyFrames(list(obs_q), stack_axis=0)
            r = sum(rewards[i : min(i + args.frame_skip, len(rewards))])

            replay_buffer.append(state=state, action=actions[i], reward=r,
                                 next_state=next_state, next_action=None,
                                 is_state_terminal=done, demo=True)

    logger.info("Buffer filled, size: %s", len(replay_buffer))


def main(argv=None):
    """Parses arguments and runs the example
    """
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env', type=str, choices=['MineRLTreechopVectorObf-v0', 'MineRLObtainDiamondVectorObf-v0',
                            'MineRLNavigateVectorObf-v0'], required=True, help='MineRL environment identifier')

    # Meta
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files. If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--logging_filename', type=str, default='log.txt', help='Filename for logging')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation')
    parser.add_argument('--only_pretrain', type=int, default=0)

    # NOT USED (from mod)

    # parser.add_argument('--adam-eps', type=float, default=1e-8, help='Epsilon for Adam.')
    # parser.add_argument('--gray-scale', action='store_true', default=False, help='Convert pov into gray scaled image.')
    # training scheme (agent)
    # parser.add_argument('--agent', type=str, default='DQN', choices=['DQN', 'DoubleDQN', 'PAL', 'CategoricalDoubleDQN'])
    # # network architecture
    # parser.add_argument('--arch', type=str, default='dueling', choices=['dueling', 'distributed_dueling'],
    #                     help='Network architecture to use.')
    # experience replay buffer related settings
    # parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
    #                     help='Minimum replay buffer size before performing gradient updates.')
    # parser.add_argument('--prioritized', action='store_true', default=False, help='Use prioritized experience replay.')

    # Update rules
    parser.add_argument('--no-clip-delta', dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--update-interval', type=int, default=4, help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--num-step-return', type=int, default=10)
    parser.add_argument('--eval-n-runs', type=int, default=3)
    parser.add_argument('--error-max', type=float, default=1.0)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 4,
                        help='Frequency (in timesteps) at which the target network is updated.')
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['rmsprop', 'adam'])
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate')
    parser.add_argument("--replay-buffer-size", type=int, default=10 ** 6,
                        help="Size of replay buffer (Excluding demonstrations)")    # --replay-capacity in mod
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument('--batch-accumulator', type=str, default="sum", choices=['sum', 'mean'],
                        help='accumulator for batch loss.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount rate.')

    # Exploration settings
    parser.add_argument('--final-exploration-frames', type=int, default=10 ** 6,
                        help='Timesteps after which we stop annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01, help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--replay-start-size', type=int, default=1000,
                        help='Minimum replay buffer size before performing gradient updates.')

    # DQfD specific parameters for loading and pretraining.
    parser.add_argument('--n-experts', type=int, default=10)
    parser.add_argument('--n-clusters', type=int, default=30, help='#clusters for K-means')
    parser.add_argument('--n-pretrain-steps', type=int, default=750000)
    parser.add_argument('--demo-supervised-margin', type=float, default=0.8)
    parser.add_argument('--loss-coeff-l2', type=float, default=1e-5)
    parser.add_argument('--loss-coeff-nstep', type=float, default=1.0)
    parser.add_argument('--loss-coeff-supervised', type=float, default=1.0)
    parser.add_argument('--bonus-priority-agent', type=float, default=0.001)
    parser.add_argument('--bonus-priority-demo', type=float, default=1.0)
    parser.add_argument('--n-pretrain-rounds', type=int, default=1)

    # NoisyNet parameters
    parser.add_argument('--use-noisy-net', type=str, default=None,
                        choices=['before-pretraining', 'after-pretraining'])
    parser.add_argument('--noisy-net-sigma', type=float, default=0.5,
                        help='NoisyNet explorer switch. This disables following options: '
                             '--final-exploration-frames, --final-epsilon, --eval-epsilon')

    # Parameters for state/action handling
    parser.add_argument('--frame-stack', type=int, default=None, help='Number of frames stacked (None for disable).')
    parser.add_argument('--frame-skip', type=int, default=None, help='Number of frames skipped (None for disable).')
    parser.add_argument('--use-full-observation', action='store_true', default=False)

    args = parser.parse_args(argv)
    args.outdir = pfrl.experiments.prepare_output_dir(args, args.outdir)


    """if args.logging_filename is not None:
        logging.basicConfig(filename=args.logging_filename, filemode='w',
                            level=args.logging_level)
    else:
        logging.basicConfig(level=args.logging_level)"""

    logger = logging.getLogger(__name__)

    # Logging not original
    log_format = '%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s'
    logging.basicConfig(filename=os.path.join(args.outdir, args.logging_filename), format=log_format, level=args.logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logging_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console_handler)  # add hander to the root logger

    # logger.info(sys.version)  # Python version
    # logger.info(','.join(freeze.freeze()))  # pip freeze

    # logger.info('Output files will be saved in {}'.format(args.outdir))

    # utils.log_versions()

    # From mod
    # os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = args.outdir
    # args.outdir = experiments.prepare_output_dir(args, args.outdir)

    # Set different random seeds for train and test envs.
    train_seed = args.seed  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - args.seed

    # Set a random seed used in ChainerRL.
    pfrl.utils.set_random_seed(args.seed)

    # K-Means
    kmeans = cached_kmeans(
        cache_dir=os.environ.get('KMEANS_CACHE'),
        env_id=args.env,
        n_clusters=args.n_clusters,
        random_state=args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    logger.info('Output files are saved in {}'.format(args.outdir))

    # if args.load:
    #     logger.info("Loading KMeans clustering")
    #     path = os.path.join(os.path.split(args.load)[0], "kmeans.pkl")
    #     with open(path, "rb") as f:
    #         kmeans = pickle.load(f)
    #     logger.info("KMeans loaded")
    # else:
    #     logger.info("Starting KMeans clustering")
    #     kmeans = get_kmeans_clustering(args, logger)
    #     path = os.path.join(args.outdir, "kmeans.pkl")
    #     with open(path, 'wb') as f:
    #         pickle.dump(kmeans, f)
    #     logger.info("Completed KMeans clustering")

    # Load / make environment
    def make_env(env, test):
        if isinstance(env, gym.wrappers.TimeLimit):
            logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
            env = env.env
            max_episode_steps = env.spec.max_episode_steps
            env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

        # NOTE: wrapping order matters!
        if args.use_full_observation:
            env = FullObservationSpaceWrapper(env)
        elif args.env.startswith('MineRLNavigate'):
            env = PoVWithCompassAngleWrapper(env)
        else:
            env = ObtainPoVWrapper(env)
        if test and args.monitor:
            env = gym.wrappers.Monitor(
                env, os.path.join(args.outdir, 'monitor'),
                mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
        if args.frame_skip is not None:
            env = FrameSkip(env, skip=args.frame_skip)

        # convert hwc -> chw as Chainer requires
        env = MoveAxisWrapper(env, source=-1, destination=0,
                              use_tuple=args.use_full_observation)
        if args.frame_stack is not None:
            env = FrameStack(env, args.frame_stack, channel_order='chw',
                             use_tuple=args.use_full_observation)

        # wrap env: action...
        env = ClusteredActionWrapper(env, clusters=kmeans.cluster_centers_)

        if test:
            env = RandomizeAction(env, args.eval_epsilon)

        env_seed = test_seed if test else train_seed
        env.seed(int(env_seed))
        return env

    logger.info('The first `gym.make(MineRL*)` may take several minutes. Be patient!')
    core_env = gym.make(args.env)
    # training env
    env = make_env(core_env, test=False) # could use wrap_env_partial instead (mod)
    # evaluation env
    eval_env = make_env(core_env, test=True) # could use wrap_env_partial instead (mod)

    # Q function / network
    if args.env.startswith('MineRLNavigate'):
        if args.use_full_observation:
            base_channels = 3  # RGB
        else:
            base_channels = 4  # RGB + compass
    elif args.env.startswith('MineRLObtain'):
        base_channels = 3  # RGB
    else:
        base_channels = 3  # RGB

    if args.frame_stack is None:
        n_input_channels = base_channels
    else:
        n_input_channels = base_channels * args.frame_stack

    q_func = DuelingDQN(args.n_clusters, n_input_channels=3 * args.frame_stack)

    # Feature extractor applied to observations
    def phi(x):
        # observation -> NN input
        if args.use_full_observation:
            pov = np.asarray(x[0], dtype=np.float32) / 255.0
            others = np.asarray(x[1], dtype=np.float32)
            return (pov, others)
        else:
            return np.asarray(x, dtype=np.float32) / 255.0

    # Specifies exploration strategy
    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(args.n_clusters))

    # calculate corresponding `steps` and `eval_interval` according to frameskip
    # 8,000,000 frames = 1333 episodes if we count an episode as 6000 frames,
    # 8,000,000 frames = 1000 episodes if we count an episode as 8000 frames.
    maximum_frames = 8000000  # = 1440 episodes if we count an episode as 6000 frames.
    if args.frame_skip is None:
        steps = maximum_frames
        eval_interval = 20000 * 10  # (approx.) every 100 episode (counts "1 episode = 6000 steps") def: 6000 * 100
    else:
        steps = maximum_frames // args.frame_skip
        eval_interval = 20000 * 10 // args.frame_skip  # (approx.) every 100 episode (counts "1 episode = 6000 steps"), def. 6000 * 100

    # Anneal beta from beta0 to 1 throughout training
    betasteps = steps / args.update_interval
    # Creates prioritized replay buffer
    replay_buffer = PrioritizedDemoReplayBuffer(
        args.replay_buffer_size, alpha=0.4,
        beta0=0.6, betasteps=betasteps,
        error_max=args.error_max,
        num_steps=args.num_step_return)

    # Fill the demo buffer with expert transitions
    if not args.demo:
        fill_buffer(replay_buffer, args, kmeans, logger)

    def reward_transform(x):
        return np.sign(x) * np.log(1 + np.abs(x))

    # Noisy net stuff
    if args.use_noisy_net is not None and args.use_noisy_net == 'before-pretraining':
        pfrl.nn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        explorer = explorers.Greedy()

    # Specifies optimizer
    if args.optimizer == 'rmsprop':
        opt = optim.RMSprop(q_func.parameters(), lr=args.lr, alpha=0.95,
                            momentum=0.0, eps=1e-2,
                            weight_decay=args.loss_coeff_l2)
    elif args.optimizer == 'adam':
        opt = optim.Adam(q_func.parameters(), lr=args.lr,
                         weight_decay=
                         args.loss_coeff_l2)

    agent = DQfD(q_func, opt, replay_buffer,
                 gamma=args.gamma,
                 explorer=explorer,
                 n_pretrain_steps=args.n_pretrain_steps,
                 demo_supervised_margin=args.demo_supervised_margin,
                 bonus_priority_agent=args.bonus_priority_agent,
                 bonus_priority_demo=args.bonus_priority_demo,
                 loss_coeff_nstep=args.loss_coeff_nstep,
                 loss_coeff_supervised=args.loss_coeff_supervised,
                 gpu=args.gpu,
                 replay_start_size=args.replay_start_size,
                 target_update_interval=args.target_update_interval,
                 clip_delta=args.clip_delta,
                 update_interval=args.update_interval,
                 batch_accumulator=args.batch_accumulator,
                 phi=phi,
                 reward_transform=reward_transform,
                 minibatch_size=args.minibatch_size)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev: {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:

        for i in range(args.n_pretrain_rounds):
            # Pre-train
            agent.pretrain()

            # Save pre-trained agent
            save_dir = args.outdir + '/saved_pretrain' + str((i + 1) * args.n_pretrain_steps)
            agent.save(save_dir)
            logger.info('Agent saved in ' + save_dir)

        if args.only_pretrain:
            # Evaluate agent
            eval_stats = experiments.eval_performance(
                env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
            logger.info('Evaluation of pretraining @ ' + str((i + 1) * args.n_pretrain_steps) + ' steps')
            logger.info('n_runs: {} mean: {} median: {} stdev: {}'.format(
                args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
        else:
            # evaluator = Evaluator(agent=agent,
            #                       n_steps=None,
            #                       n_episodes=args.eval_n_runs,
            #                       eval_interval=eval_interval,
            #                       outdir=args.outdir,
            #                       max_episode_len=None,
            #                       env=eval_env,
            #                       step_offset=0,
            #                       save_best_so_far_agent=True,
            #                       logger=logger)
            #
            # # Evaluate the agent BEFORE training begins
            # evaluator.evaluate_and_update_max_score(t=0, episodes=0)
            #
            # experiments.train_agent(agent=agent,
            #                         env=env,
            #                         steps=steps,
            #                         outdir=args.outdir,
            #                         max_episode_len=None,
            #                         step_offset=0,
            #                         evaluator=evaluator,
            #                         successful_score=None,
            #                         step_hooks=[],
            #                         checkpoint_freq=True) ###TODO: check

            # Continue training
            experiments.train_agent_with_evaluation(
                agent=agent, env=env, steps=steps,
                eval_n_steps=None, eval_n_episodes=args.eval_n_runs, eval_interval=eval_interval,
                outdir=args.outdir, eval_env=eval_env, save_best_so_far_agent=True)

    env.close()
    eval_env.close()
    # aicrowd_helper.register_progress(1)


if __name__ == "__main__":
    main()
