import datetime
import os
import argparse
import torch

from rlpyt.samplers.collections import TrajInfo
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper
from dreamer.utils.configs import configs, load_configs

from evaluator import Evaluator

def build_and_train(log_dir, game="cartpole_balance", run_ID=0, cuda_idx=None, eval=False, save_model='last', load_model_path=None, 
        sample_rand=1, rand_iter=100000):
    domain, task = game.split('_',1)
    if '_' in task:
        d,task=task.split('_')
        domain+='_'+d
    
    params = torch.load(load_model_path) if load_model_path else {}

    if load_model_path is not None:
        load_dir = os.path.dirname(load_model_path)
    else:
        load_dir = None
    load_configs(load_dir=load_dir, save_dir=log_dir)
        
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    action_repeat = configs.action_repeat
    factory_method = make_wapper(
        DeepMindControl,
        [ActionRepeat, NormalizeActions, TimeLimit],
        [dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])
    sampler = SerialSampler(
        EnvCls=factory_method,
        TrajInfoCls=TrajInfo,
        env_kwargs=dict(name=game),
        eval_env_kwargs=dict(name=game),
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    agent = DMCDreamerAgent(train_noise=0.3, eval_noise=0, expl_type="additive_gaussian",
                              expl_min=None, expl_decay=None, initial_model_state_dict=agent_state_dict, 
                              sample_rand=sample_rand, rand_iter=rand_iter,
                              model_kwargs={"cuda_idx": cuda_idx, "domain": domain, "task": task})
    
    evaluator=Evaluator(agent, factory_method(name=game))
    algo = Dreamer(evaluator, initial_optim_state_dict=optimizer_state_dict)  # Run with defaults.
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=5e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dreamer_" + game
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    parser.add_argument('--br', help='branch name', default=None)
    parser.add_argument('--cuda-idx', help='gpu to use ', type=int, default=None)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save-model', help='save model', type=str, default='last',
                        choices=['all', 'none', 'gap', 'last'])
    
    parser.add_argument('--sample-rand', help='between 0 and 1', type=float, default=1)
    parser.add_argument('--rand-iter', type=int, default=100000)
    
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)

    parser.add_argument('--load-model-path', type=str, default=None)
    default_log_dir = os.path.join(
        os.path.dirname(__file__),
        'data',
        'local',
        datetime.datetime.now().strftime("%Y%m%d"))

    parser.add_argument('--log-dir', type=str, default=default_log_dir)
    args = parser.parse_args()

    if args.br is not None:
        args.log_dir=os.path.join(
            os.path.dirname(__file__),
            'data',
            'local',
            args.br)

    log_dir = os.path.abspath(args.log_dir)
    
    i = args.run_ID
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        print(f'run {i} already exists. ')
        i += 1
    print(f'Using run id = {i}')
    args.run_ID = i

    if args.br is not None and i > 0:
        args.load_model_path =os.path.join(log_dir, 'run_'+str(i-1), 'params.pkl')

    build_and_train(
        log_dir,
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        eval=args.eval,
        save_model=args.save_model,
        load_model_path=args.load_model_path,
        sample_rand=args.sample_rand,
        rand_iter=args.rand_iter
        )
