# View-Semantic Fisher Contrastive Learning for View-Invariant Skeleton-based Action Recognition(VS-FCL)
This text is used to explain the code involved in our work in this paper.

## Implementations
The implementation is based on Python 3.6 and Pytorch 1.7.1.
We recommend you use conda to install the dependencies.
All the requirements are listed bellow:
* os
* time
* torch
* numpy 
* tensorboard_logger
* apex 
* h5py
* sys
* random
* argparse
## Training
### stage1:
You can train your own model by running the training file:
The aim of the first-stage training is to eliminate differences between various viewpoints. Therefore, we restrict the input data to be of the same category.
Noting that because the ST-CVR learning, if you change your CUDA_VISIBLE_DEVICES=0, then the batch_size should be changed to 16.
```
CUDA_VISIBLE_DEVICES=0,1,2,3  python train_stage1.py --model ctrgcn --batch_size 64 --num_workers 0 --model_path results/ --case 0 --dataset UESTC --tb_path tb_logger/ --amp --resume plt_model/ckpt_epoch_30.pth
```
Also you can resume your training process as following:
```
CUDA_VISIBLE_DEVICES=0,1,2,3  python train_stage1.py --model ctrgcn --batch_size 64 --num_workers 0 --model_path results/ --case 0 --dataset UESTC --tb_path tb_logger/ --amp --resume plt_model/ckpt_epoch_30.pth
```

### stage2:
After the training of stage1, The V-FCL assists the view-common representation learning, which heavily reduces the influence of view change.Thus S-FCL is applied on the view-common action featurefc for semantic disentanglement. You can train your own model by running the training file:
```
CUDA_VISIBLE_DEVICES=0,1,2,3  python train_stage2.py --model ctrgcn --batch_size 64 --num_workers 0 --model_path results/ --case 0 --dataset UESTC --tb_path tb_logger/ --amp --pretrain_path [your saved checkpoint of stage1]
```

## Test
You can test your model, by running:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --model ctrgcn --batch_size 64 --num_workers 0 --model_path results/stage2_lossquan_memory_nce_16384_ctrgcn_lr_0.01_decay_0.0001_bsz_64_case_1_dataset_UESTC_amp_O2_view_Lab/ckpt_epoch_42.pth  --case 1 --dataset UESTC --tb_path Test_tb_logger/ --amp --state test
```


We provide the backbone model in models. The dataset_new_stage1.py and dataset_new_stage2.py represent the different dataloader function in stage1 training and stage 2 trainging respectively. Also, the files ctrgcn1_stage1.py and ctrgcn_stage2.py in the models folder represent two different backbone networks for two separate training stages, each with slight modifications.

