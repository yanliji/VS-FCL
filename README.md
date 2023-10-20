# View-Semantic Fisher Contrastive Learning for View-Invariant Skeleton-based Action Recognition(VS-FCL)

This code is part of the paper: *View-Semantic Fisher Contrastive Learning for View-Invariant Skeleton-based Action Recognition*
***
The view change is a serious challenge for extracting invariant representation for action analysis due to occlusion and deformation. To address this problem, we propose a View-Semantic Fisher Contrastive Learning (VS-FCL) for view-invariant action representation and recognition. The VS-FCL consists of two components, View-term Fisher Contrastive Learning (V-FCL) and Semantic-term Fisher Contrastive Learning (S-FCL), where V-FCL propels view disentanglement for obtaining view-invariant action representation and S-FCL drives the semantic disentanglement to seek effective semantic-oriented representation for accurate action
recognition. Besides, we introduce the Spatio-Temporal Cross-View Representation (ST-CVR) learning to capture view-interactive action features to fuse action information from different views, so as to guarantee obtaining view-invariant representation and improving recognition accuracy. Extensive and fair
evaluations are conducted on the UESTC, NTU 60, NTU 120, and Northwestern-UCLA datasets. The experiment results show that our proposed approach achieves outstanding performance on all datasets for view-invariant action recognition.
***   
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
##  View-Semantic Fisher Contrastive Learning (VS-FCL) Results
The VS-FCL results include two sub-parts: View-term FCL (V-FCL) and Semantic-term FCL (S-FCL).

The V-FCL is presented to assist view disentanglement for generating view-common
and view-specific action representations.Bring closer the common viewpoint features of the same category, and separate the unique viewpoint features of the same category.

Base on the V-FCL, The Semantic-term FCL (S-FCL) propels semantic disentanglement to learn semantic-oriented action representations.

### Visualization
Here, we  provide the t-SNE visualization of feature distributions learned by the VS-FCL algorithm in the UESTC dataset. Different colors and markers represent different actions and viewpoints.Specifically, we provide the original viewpoint features, the view-common features obtained after V-FCL learning, and the semantic features obtained after VS-FCL learning. Our goal is to eliminate the impact of multi-view differences through the learning of the first-stage V-FCL, thereby promoting the separation of semantic features of various categories in the second-stage VS-FCL.
<!-- <figure class="third">
    <img src="./bigjpg/f-c-3_1.jpg">
    <figcaption>这是图片1的标题</figcaption>
    <img src="./bigjpg/f-c-3_2.jpg">
    <figcaption>这是图片2的标题</figcaption>
    <img src="./bigjpg/f-c-3_3.jpg">
    <figcaption>这是图片3的标题</figcaption>
</figure> -->
<!-- - Sample 1：Distribution of original action representation

<img src="./bigjpg/L-S-fv-40_1_begin.png" width=220><img src="./bigjpg/L-S-fv-40_2_begin.png" width=220><img src="./bigjpg/L-S-fv-40_3_begin.png" width=220>

 - Sample 2：View-common representation obtained by V-FCL(3 classes)

<img src="./bigjpg/f-c-3_1.jpg" width=220><img src="./bigjpg/f-c-3_2.jpg" width=220><img src="./bigjpg/f-c-3_3.jpg" width=220> -->

<!-- |View-common Representation|<img src="./bigjpg/f-c-3_1.jpg" width="200">|<img src="./bigjpg/f-c-3_2.jpg" width="200">|<mg src="./bigjpg/f-c-3_3.jpg" width="200">
|:-:|:-:|:-:| -->
- Sample 1：View-origin representation(3 classes)

<img src="./imgs/f-v-3_1.jpg" width=220><img src="./imgs/f-v-3_2.jpg" width=220><img src="./imgs/f-v-3_3.jpg" width=220>


- Sample 2：View-common representation obtained by V-FCL(3 classes)

<img src="./imgs/V-FCL-3_1.jpg" width=220><img src="./imgs/V-FCL-3_2.jpg" width=220><img src="./imgs/V-FCL-3_3.jpg" width=220>

- Sample 3：Semantic-oriented representationobtained by VS-FCL(3 classes)

<img src="./imgs/VS-FCL-3_1.jpg" width=220><img src="./imgs/VS-FCL-3_2.jpg" width=220><img src="./imgs/VS-FCL-3_3.jpg" width=220>

- Sample 4：View-origin representation(10 classes)

<img src="./imgs/F-V-10_1.jpg" width=220><img src="./imgs/F-V-10_2.jpg" width=220><img src="./imgs/F-V-10_3.jpg" width=220>

- Sample 5：View-common representation obtained by V-FCL(10 Classes)

<img src="./imgs/V-FCL-10_1.jpg" width=220><img src="./imgs/V-FCL-10_2.jpg" width=220><img src="./imgs/V-FCL-10_3.jpg" width=220>

- Sample 6：Semantic-oriented representationobtained by VS-FCL(10 classes)

<img src="./imgs/VS-FCL-10_1.jpg" width=220><img src="./imgs/VS-FCL-10_2.jpg" width=220><img src="./imgs/VS-FCL-10_3.jpg" width=220>
<!-- - Sample 5: View-specific representation obtained by V-FCL(1 class)

<!-- <img src="./bigjpg/fs_17_1.jpg" width=220><img src="./bigjpg/fs_17_2.jpg" width=220><img src="./bigjpg/fs_17_3.jpg" width=220> --> 



## Conclusion
In this paper, we have proposed a View-Semantic Fisher Contrastive Learning (VS-FCL) algorithm, which designs the V-FCL and S-FCL to drive view and semantic disentanglement and obtained view-invariant semantic-oriented action representation for correct recognition, efficiently dealing with the view change problem. Four large-scale datasets were adopted to evaluate the proposed VS-FCL algorithm. Comparison with SOTAs sufficiently certified the superiority of our VS-FCL for view-invariant action recognition. Ablation studies on VS-FCL and ST-CVR components further
certified their solid contributions for view-invariant action representation and recognition.
***

## Citation
```
@inproceedings{gao2022global,
  title={Global-local cross-view fisher discrimination for view-invariant action recognition},
  author={Gao, Lingling and Ji, Yanli and Yang, Yang and Shen, HengTao},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={5255--5264},
  year={2022}
}
```
