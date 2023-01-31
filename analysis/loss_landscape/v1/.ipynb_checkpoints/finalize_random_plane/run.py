from essl.utils import ll_random_plane
import os
import glob
DISTANCE = 10
STEPS = 50
LANDSCAPE_DIRNAME = "loss_landscapes_2"
# EXP_DIRS = glob.glob("/home/noah/ESSL/final_exps/optimization/*6*/*")
EXP_DIRS = [
            #'/home/noah/ESSL/final_exps/optimization/exp8_7/7',
            #'/home/noah/ESSL/final_exps/optimization/exp8_6/1',
            # '/home/noah/ESSL/final_exps/optimization/exp8_5/2',
            # '/home/noah/ESSL/final_exps/optimization/exp8_4/4'
            # '/home/noah/ESSL/final_exps/optimization/exp6_0/3'
            '/home/noah/ESSL/final_exps/optimization/exp6_1/2'
]
if __name__ == "__main__":
    for exp in EXP_DIRS:
        print(f"running ll on {exp}")
        try:
            model_path = glob.glob(os.path.join(exp, "models/*.pt"))
            if len(model_path) == 1:
                model_path = model_path[0]
            else:
                model_path = [m for m in model_path if "downstream" in m][0]
            ll_dir = os.path.join(exp, LANDSCAPE_DIRNAME)
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

