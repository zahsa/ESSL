# Pretext task Optimization  

This repository includes main source codes for our paper about "Pretext Task Learning and Optimization".
Our objective is to learn the parameters including in the first phase of Self-Supervised Learning and to optimize the pretext task for different datasets and problems.

## Dependencies and Installation

Installation from source

```
cd path/to/ESSL
pip install -e .
```

## Running an Experiment using CLI

To run a training experiment using the CLI, after installation, one can use the essl_train command.

```
essl_train \
  --pop_size 2 \
  --num_generations 2 \
  --cxpb 0.2 \
  --mutpb 0.5 \
  --dataset Cifar10 \
  --backbone tinyCNN_backbone \
  --ssl_task SimCLR \
  --ssl_epochs 5 \
  --ssl_batch_size 256 \
  --evaluate_downstream_method finetune \
  --device cuda \
  --exp_dir ./ \
  --use_tensorboard True \
  --save_plots True
```
## Citation
```bibtex
@article{barrett2023evolutionary,
  title={Evolutionary Augmentation Policy Optimization for Self-supervised Learning},
  author={Barrett, Noah and Sadeghi, Zahra and Matwin, Stan},
  journal={AAIML.2023.1167},
  year={2023}
}
