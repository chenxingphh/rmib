import os
import sys
import json5
from pprint import pprint
from src.utils import params
from src.trainer import Trainer
import torch
import numpy as np
import random


def seed_torch(seed=32):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    argv = sys.argv
    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])

        for args, config in arg_groups:
            seed_torch(args.seed)

            trainer = Trainer(args)
            states = trainer.train()
            with open('models/log.jsonl', 'a') as f:
                f.write(json5.dumps({
                    'data': os.path.basename(args.data_dir),
                    'params': config,
                    'state': states,
                }))
                f.write('\n')
    elif len(argv) == 3 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print('Usage: "python train.py configs/main.json5"')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["TOKENIZERS_PARALLELISM"] = 'true'
    main()
