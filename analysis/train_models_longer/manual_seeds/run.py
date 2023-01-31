from essl.utils import train_model_longer
import os
import glob
def get_algorithm(params_path):
    with open(params_path, "r") as f:
        params = []
        for i, l in enumerate(f.readlines()):
            params.append(l)
    # get params
    ssl = [j.split(" ")[1].strip("\n") for j in params if "ssl_task" in j][0]
    return ssl

def run_exp(save_dir, train_kwargs, ssl_epochs= [40, 90, 990], downstream_epochs= [50]):
    
    pretext_model_path = glob.glob(os.path.join(save_dir, "models/*_pretext.pt"))
    if len(pretext_model_path) == 0:
        return
    else:
        pretext_model_path = pretext_model_path[0]
    outcome_path = os.path.join(save_dir, "outcomes.json")
    algo = get_algorithm(os.path.join(save_dir, "params.txt"))
    for se in ssl_epochs:
        for de in downstream_epochs:
            exp_dir = os.path.join(save_dir, f"{se}_{de}")
            if not os.path.isdir(exp_dir):
                os.mkdir(exp_dir)

            print(f"algorithm: {algo}")
            print(f"dataset: {train_kwargs['dataset']}")
            print(f"training ssl e: {se}")
            print(f"training downstream e: {de}")
            train_model_longer(
                pretext_model_path=pretext_model_path,
                outcome_path=outcome_path,
                backbone="largerCNN_backbone",
                ssl_task=algo,
                ssl_epochs=se,
                downstream_epochs=de,
                save_dir=exp_dir,
                **train_kwargs)
# # cifar10
# b256 = glob.glob("/home/noah/ESSL/final_exps/optimization/exp8*/*")
b256 = ["/home/noah/ESSL/final_exps/optimization/exp8_4/4"]
b256_params = {"dataset":"Cifar10", "ssl_batch_size":256}
# run
for e in b256:
    run_exp(e, b256_params, ssl_epochs=[990])
# b32 = glob.glob("/home/noah/ESSL/final_exps/optimization/exp6*/*")
# b32_params = {"dataset":"Cifar10", "ssl_batch_size":32}
# # run
# for e in b32:
#     run_exp(e, b32_params)

# SVHN
# b256_svhn = glob.glob("/home/noah/ESSL/final_exps/optimization/exp10*/*")
b256_svhn = [
    ["/home/noah/ESSL/final_exps/optimization/exp10_3/3", [40, 90]],
    ["/home/noah/ESSL/final_exps/optimization/exp10_1/4", [990]]
]
b256_params_svhn = {"dataset":"SVHN", "ssl_batch_size":256}
# run
for e, ssl_epochs in b256_svhn:
    run_exp(e, b256_params_svhn, ssl_epochs=ssl_epochs)

b32_svhn = glob.glob("/home/noah/ESSL/final_exps/optimization/exp11*/*")
b32_svhn = [
    ["/home/noah/ESSL/final_exps/optimization/exp11_0/2", [40, 90]],
    ["/home/noah/ESSL/final_exps/optimization/exp11_1/0", [990]]
]
b32_params_svhn = {"dataset":"SVHN", "ssl_batch_size":32}
# run
for e, ssl_epochs in b32_svhn:
    run_exp(e, b32_params_svhn, ssl_epochs=ssl_epochs)
