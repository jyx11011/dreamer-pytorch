import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dreamer.utils.configs import configs,load_configs,attributes

def print_configs(configs):
    for atrr in attributes:
        v=getattr(configs, atrr, None)
        print(atrr+': '+str(v), end=', ')

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--br', type=str, default=None)
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--it', type=str, default=None)

    args = parser.parse_args()

    path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'local',
        args.br,
        'run_'+args.run,
        'eval_log',
        'run_'+args.it
        )
    load_configs(load_dir=path)
    print_configs(configs)
    i=0

    print() 
    f, s = plt.subplots(2,figsize=(10,6))
    while os.path.exists(os.path.join(path, 'iter_' + str(i)+'.npz')):
        f=os.path.join(path, 'iter_' + str(i)+'.npz')
        data=np.load(f,allow_pickle=True)

        pos=list(map(lambda x: x['position'][0], data['observations']))
        s[0].plot(pos,label='run_'+str(i))
        s[0].set(xlabel='Timestep',ylabel='x')
        theta=list(map(lambda x:np.arctan(x['position'][2]/x['position'][1]) , data['observations']))
        s[1].plot(theta,label='run_'+str(i))
        s[1].set_ylim([-np.pi, np.pi])
        s[1].set(xlabel='Timesteps',ylabel='theta')
        print(len(data['observations']))
        i+=1
    p=os.path.join(path, 'plt.png')
    a=['timesteps', 'action_repeat', 'stochastic_size_state_size', 'deterministic_size'] 
    t=''.join([x+': '+getattr(configs,x) for x in a])
    plt.title(t)
    plt.show()
    plt.savefig(p)


        
