import glob
import os

if __name__ == "__main__":
    imp="/home/noah/ESSL/exps/Analysis/importance_plots2"
    sens="/home/noah/ESSL/exps/Analysis/sensitivity_plots2"
    exps = {"exp6":"c10_bs32",
            "exp8":"c10_bs256",
            "exp10":"svhn_bs256",
            "exp11":"svhn_bs32"}
    for i in glob.glob(os.path.join(sens, "*")):
        
        f = i.split("/")
        fig = f[-1]
        exp = exps[fig.split("_")[0]]
        fig = exp + "_"+ fig.split("_")[-1]
        f[-1] = fig
        fig = "/".join(f)
        print(i)
        print(fig)
        os.rename(i, fig)