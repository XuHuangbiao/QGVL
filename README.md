# Quality-Guided Vision-Language Learning for Long-Term Action Quality Assessment

This is the code for TMM2025 paper "Quality-Guided Vision-Language Learning for Long-Term Action Quality Assessment".

## Environments

- Tesla P100
- CUDA: 10.2
- Python: 3.7.15
- PyTorch: 1.12.0+cu102

## Features

The features and label files of Rhythmic Gymnastics and Fis-V dataset can be download from the [GDLT](https://github.com/xuangch/CVPR22_GDLT) repository.

The features and label files of FS1000 dataset can be download from the [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) repository.

The features and label files of FineFS dataset can be download from the [FineFS](https://github.com/yanliji/FineFS-dataset) repository.

## Pretrained Model
If you wish to extract your own action text labels, please download the [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) pretrained model and place it in:
```
weights/k400_clip_complete_finetuned_30_epochs.pth
```

## Running

Please fill in or select the args enclosed by {} first.

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {Ball/Clubs/Hoop/Ribbon} --lr 1e-2 --epoch {250/400/500/150} --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {Ball/Clubs/Hoop/Ribbon} --n_decoder 2 --n_query 4 --dropout 0.3 --test --ckpt {the name of the used checkpoint}
```

## Important Notes
In the current version, the best performance is achieved when the number of score-level quality patterns is set to 101 (i.e., a score interval of 1). This configuration fully exploits the fine-grained discriminability of score differences. However, it also greatly increases the optimization difficulty — the distinctions between adjacent score patterns become extremely subtle, which may lead to gradient instability or pattern collapse. Therefore, achieving the best performance requires strict adherence to the original reproduction environment and experimental settings.

If your model training fails to converge or fit your application well, you may consider reducing the number of score-level quality patterns, such as to 21 (i.e., a score interval of 5). This adjustment will significantly ease optimization while still providing competitive performance.

## Citation
If our project is helpful for your research, please consider citing:
```
@article{xu2025quality,
  title={Quality-Guided Vision-Language Learning for Long-Term Action Quality Assessment},
  author={Xu, Huangbiao and Wu, Huanqi and Ke, Xiao and Li, Yuezhou and Xu, Rui and Guo, Wenzhong},
  journal={IEEE Transactions on Multimedia},
  volume={27},
  pages={7326--7339},
  year={2025}
}
```

## Acknowledgement
This repository builds upon [GDLT (CVPR 2022)](https://github.com/xuangch/CVPR22_GDLT).

We thank the authors for their contributions to the research community.
