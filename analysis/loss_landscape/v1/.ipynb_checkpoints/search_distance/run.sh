export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0

essl_ll_random_plane \
  --model_path "/home/noah/ESSL/final_exps/optimization/exp8_6/1/models/119_downstream.pt" \
  --dataset "Cifar10" \
  --backbone "largerCNN_backbone" \
  --save_dir "/home/noah/ESSL/exps/Analysis/loss_landscape/search_distance/100_100" \
  --distance 100 \
  --steps 100

essl_ll_random_plane \
  --model_path "/home/noah/ESSL/final_exps/optimization/exp8_6/1/models/119_downstream.pt" \
  --dataset "Cifar10" \
  --backbone "largerCNN_backbone" \
  --save_dir "/home/noah/ESSL/exps/Analysis/loss_landscape/search_distance/500_500" \
  --distance 500 \
  --steps 500

