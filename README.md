# IRAOD-New Environment Setup

This document only records the steps needed to create the project environment
and install the required packages.

## 1. Enter the Project

```bash
cd /home/storageSDA1/liaojr/IRAOD-New
```

## 2. Create the Conda Environment

```bash
conda create -n iraod python=3.10 -y
conda activate iraod
```

## 3. Install PyTorch

Install the CUDA 11.8 build used by this project:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 4. Install Project Dependencies

Install all Python packages from the project requirements file:

```bash
pip install -r /home/storageSDA1/liaojr/IRAOD-New/requirements.txt
```

## 5. Verify the Environment

```bash
python test_sarclip.py
```

## 6. Run Training

Raw self-training on RSAR chaff:

```bash
python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_rsar1.py --cfg-options corrupt="chaff"
```

SARCLIP CGA self-training on RSAR chaff:

```bash
CGA_SCORER=sarclip python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_rsar1.py --cfg-options corrupt="chaff"
```
