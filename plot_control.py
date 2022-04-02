import os
import argparse
import numpy as np
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
    while os.path.exists(os.path.join(path, 'iter_' + str(i)+'.npz')):
        f=os.path.join(path, 'iter_' + str(i)+'.npz')
        data=np.load(f,allow_pickle=True)
        print(data['observations'])
        print(data['actions'])
        i+=1


        
