export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1

essl_GA_mo_bootstrap \
  --pop_size 15 \
  --num_generations 10 \
  --cxpb1 0.8 \
  --mutpb1 0.8 \
  --cxpb2 0.5 \
  --mutpb2 0.2 \
  --dataset Cifar10 \
  --backbone largerCNN_backbone \
  --ssl_epochs 10 \
  --ssl_batch_size 32 \
  --evaluate_downstream_method finetune \
  --device cuda \
  --exp_dir /home/noah/ESSL/exps/iteration5/exp4_0 \
  --use_tensorboard True \
  --save_plots True \
  --crossover PMX \
  --chromosome_length 3 \
  --selection roulette \
  --num_seeds 3 \
  --adaptive_pb1 AGA \
  --eval_method "best val test" \
  --ssl_tasks v6 \
  --num_elite 2