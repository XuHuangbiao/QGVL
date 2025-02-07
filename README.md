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

