import glob
import os
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reformat exp dirs')
    parser.add_argument('--exp_dir', default='', help='dir that contains all the exps with runexp* format')
    parser.add_argument('--targ_dir', default='', help='new target dir')
    args = parser.parse_args()
    exps = glob.glob(os.path.join(args.exp_dir, "*exp*"))
    for e in exps:
        seed_directories = [d for d in glob.glob(os.path.join(e, "*/")) if "tensorboard" not in d and "datasets" not in d]
        for dir in seed_directories:
            new_dir = os.path.join(args.targ_dir, dir.split(args.exp_dir)[1][1:])
            shutil.move(dir, new_dir)
