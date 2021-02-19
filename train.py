import argparse
import os
import shutil
from src.engine import Engine
from src.utils.util import load_log, mkdir_p

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="Path to a config")
    parser.add_argument('--tag', default='',
                        help="tag to discern training instances")
    parser.add_argument('--train_dir',
                        help="Path to a train dir")
    args = parser.parse_args()

    train_dir = mkdir_p(args.train_dir)
    train_dir_name = train_dir.split('/')[-1]
    log_name = train_dir_name
    log = load_log(log_name)

    shutil.copyfile(args.config_path, os.path.join(train_dir, "config.yml"))

    engine = Engine(
        mode='train', config_path=args.config_path, log=log, train_dir=train_dir)
    engine.train()
