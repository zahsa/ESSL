from essl.utils import loss_landscape_analysis
import os

PARAMS = {
    "bs": [4096, 1024, 256],
    "d":[1,10,1000],
    "s":[10,50,100]
}
if __name__ == "__main__":
    model = "/home/noah/ESSL/final_exps/optimization/exp8_2/1/models/54.pt"
    save_dir = "/home/noah/ESSL/exps/Analysis/loss_landscape/search_params"
    for bs in [4096, 1024, 256]:
        for d in [1,10,1000]:
            # for s in [10,50,100]:
            s = 50
            exp_dir = os.path.join(save_dir, f"{bs}_{d}_{s}")
            if not os.path.isdir(exp_dir):
                os.mkdir(exp_dir)
            else:
                continue
            loss_landscape_analysis(model_path=model,
                                      dataset="Cifar10",
                                      backbone="largerCNN_backbone",
                                      save_dir=os.path.join(save_dir, exp_dir),
                                        batch_size=bs,
                                        distance=d,
                                        steps=s,
                                      device="cpu")