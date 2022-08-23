export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1


essl_GA_bootstrap \
  --pop_size 15 \
  --num_generations 10 \
  --cxpb 0.8 \
  --mutpb 0.8 \
  --dataset Cifar10 \
  --backbone largerCNN_backbone \
  --ssl_task SimSiam \
  --ssl_epochs 10 \
  --ssl_batch_size 32 \
  --evaluate_downstream_method finetune \
  --device cuda \
  --exp_dir /home/noah/ESSL/exps/iteration4/exp6_3 \
  --use_tensorboard True \
  --save_plots True \
  --crossover PMX \
  --chromosome_length 3 \
  --selection roulette \
  --num_seeds 3 \
  --adaptive_pb AGA \
  --eval_method "best val test" \
  --num_elite 2