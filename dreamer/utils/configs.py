import pickle
import os

attributes=['action_repeat', 
            'model_lr', 
            'stochastic_size', 'deterministic_size', 'hiddent_size', 
            'timesteps', 'iter', 'max_linesearch_iter', 'linesearch_decay',
            'eps', 'detach_unconverged', 'backprop', 'delta_u']

class Configs:
    def __init__(self, args = None):
        self.action_repeat=2

        self.model_lr=6e-4

        self.stochastic_size=1
        self.deterministic_size=5
        self.hidden_size=5

        self.timesteps=10
        self.iter=50
        self.max_linesearch_iter=20
        self.linesearch_decay=0.2
        self.eps=1e-6
        self.detach_unconverged=True
        self.backprop=False
        self.delta_u=None

        if args is not None:
            for attr in attributes:
                if hasattr(args, attr) and getattr(args, attr, None) is not None:
                    setattr(self, attr, getattr(args, attr))

    def update(self, args):
        for attr in attributes:
            if hasattr(args, attr) and getattr(args, attr, None) is not None:
                setattr(self, attr, getattr(args, attr))

    def save(self, dir):
        path=os.path.join(dir, 'configs.pkl')
        f=open(path, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    

def load_configs(load_dir = None, save_dir = None):
    if load_dir is not None:
        path=os.path.join(load_dir, 'configs.pkl')
        if os.path.exists(path):
            f = open(path, 'wb')
            configs = pickle.load(f)
        else:
            configs = Configs()
    else:
        configs = Configs()
    if save_dir is not None:
        configs.save(save_dir)

configs = Configs()
