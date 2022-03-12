from dm_control import suite
import numpy as np
import argparse

def save_goal_state(domain="cartpole", task="balance"):
    env = suite.load(domain_name=domain, task_name=task)
    camera = dict(quadruped=2).get(domain, 0)
    f=domain+"/"+domain+"_"+task
    obs=env.physics.render(64,64,camera_id=camera).transpose(2, 0, 1)
    print(env.task.get_observation(env.physics))
    np.save(f,obs)

if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    game=parser.parse_args().game
    domain, task = game.split('_')
    save_goal_state(domain, task)

#export MUJOCO_GL=egl
