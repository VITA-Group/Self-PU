# [Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training](https://arxiv.org/abs/2006.11280)
[ICML2020] Xuxi Chen*, Wuyang Chen*, Tianlong Chen, Ye Yuan, Chen Gong, Kewei Chen, Zhangyang Wang

# Overview
We proposed Self-PU Framework that introduces self-paced, self-calibrated and self-supervised learning to the PU field. 

# Method
- Self-paced learning: We gradually selected confident samples from the unlabeled set and assign labels to them. 
- Self-calibrated learning: Find optimal weights for unlabeled samples in order to obtain more source of supervision.
- Self-supervised learning: Fully exploit the learning ability of models by teacher-student structure. 
# Set-up
## Environment
```
conda install pytorch==0.4.1 cuda92 torchvision -c pytorch
conda install matplotlib scikit-learn tqdm
pip install opencv-python
```
## Preparing Data
Download cifar-10 and extract it into `cifar/`. 

# Evaluation
## Pretrained Model
MNIST: [Google Drive](https://drive.google.com/file/d/1RjVAIv_zPvKraLiyh8Oeshifun4zkgrm/view?usp=sharing "Google Drive"),
Accuracy: 94.45%

CIFAR-10: [Google Drive](https://drive.google.com/file/d/1Ybzaph0355FYjxFlPorrJBiESo_6LfJC/view?usp=sharing "Google Drive"), Accuracy: 90.05%

## Evaluation Code
MNIST:
```python
python evaluation.py --model mnist.pth.tar 
```

CIFAR-10:
```python
python evaluation.py --model cifar.pth.tar --datapath cifar --dataset cifar
```

# Training
## Baseline
### MNIST
```python
python train.py --self-paced False --mean-teacher False 
```

### CIFAR-10
```python
python train.py --self-paced False --mean-teacher False --dataset cifar --datapath cifar
```
## Self-PU (without self-calibration)
Training with self-calibation would be expensive. A cheap alternative:
### MNIST

```python
python train_2s2t.py --soft-label
```
### CIFAR-10
```python
python train_2s2t.py --dataset cifar --datapath cifar --lr 5e-4 --soft-label
```

## Self-PU 
Training with self-calibation would be expensive. A cheap alternative:
### MNIST

```python
python train_2s2t.py --soft-label
```
### CIFAR-10
```python
python train_2s2t.py --dataset cifar --datapath cifar --lr 5e-4 --soft-label
```

# Results
## Reproduce



