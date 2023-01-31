export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0

essl_ll_random_plane \
  --model_path "/home/noah/ESSL/final_exps/optimization/exp8_4/4/models/86_downstream.pt" \
  --dataset "Cifar10" \
  --backbone "largerCNN_backbone" \
  --save_dir "/home/noah/ESSL/final_exps/optimization/exp8_4/4/loss_landscapes_2" \
  --distance 10 \
  --steps 50

essl_ll_random_plane \
  --model_path "/home/noah/ESSL/final_exps/optimization/exp6_0/3/models/12.pt" \
  --dataset "Cifar10" \
  --backbone "largerCNN_backbone" \
  --save_dir "/home/noah/ESSL/final_exps/optimization/exp6_0/3/loss_landscapes_2" \
  --distance 10 \
  --steps 50

