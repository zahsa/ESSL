from essl.utils import ll_random_plane
import os
import glob
DISTANCE = 10
STEPS = 50
LANDSCAPE_DIRNAME = "loss_landscapes_2"
# EXP_DIRS = glob.glob("/home/noah/ESSL/final_exps/optimization/*6*/*")
EXP_DIRS = [
    "/home/noah/ESSL/final_exps/optimization/exp10_0/1",
    "/home/noah/ESSL/final_exps/optimization/exp10_1/4",
    "/home/noah/ESSL/final_exps/optimization/exp10_2/1",
    "/home/noah/ESSL/final_exps/optimization/exp10_3/3"
    "/home/noah/ESSL/final_exps/optimization/exp11_1/0",
    "/home/noah/ESSL/final_exps/optimization/exp11_3/1"]


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

