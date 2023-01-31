from essl.utils import ll_random_plane, ll_linear_interpolation
import os
import glob
import pandas as pd

DISTANCE = 10
STEPS = 50
SAVE_DIR = "/home/noah/ESSL/exps/Analysis/loss_landscape/v2"
best_chromos = pd.read_csv("/home/noah/ESSL/exps/Analysis/Final Results/best_chromos/best_chromos.csv")
convert = {
    "exp6":"c10_bs32",
    "exp8":"c10_bs256",
    "exp10":"svhn_bs256",
    "exp11":"svhn_bs32"
}
if __name__ == "__main__":
    for _, bc in best_chromos.iterrows():
        exp = bc.dir
        print(f"running ll on {exp}")
        try:
            model_path = glob.glob(os.path.join(exp, "models/*.pt"))
            if len(model_path) == 1:
                model_path = model_path[0]
            else:
                model_path = [m for m in model_path if "downstream" in m][0]
            ll_dir = os.path.join(SAVE_DIR, f"{convert[bc.exp.split('_')[0]]}_{bc.algo}")
            if not os.path.isdir(ll_dir):
                os.mkdir(ll_dir)
            try:
                ll_random_plane(model_path=model_path,
                                          dataset="Cifar10",
                                          backbone="largerCNN_backbone",
                                          save_dir=ll_dir,
                                            distance=DISTANCE,
                                            steps=STEPS)
            except:
               continue
            try:
                ll_linear_interpolation(model_path,
                                        dataset="Cifar10",
                                        backbone="largerCNN_backbone",
                                        save_dir=ll_dir,
                                        steps=STEPS
                                        )
            except:
                continue
        except IndexError:
            continue