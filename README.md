# 🌿 Leaf Disease Detection

A machine learning pipeline for detecting plant leaf diseases from images.  
Using preprocessing, model training, and visualization, the repository aims to provide an effortless workflow for disease classification in leaves.

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Download or Prepare Data](#download-or-prepare-data)
  - [Run Training](#run-training)
  - [Test Model](#test-model)
- [Project Structure](#project-structure)
- [Model & Evaluation](#model--evaluation)
- [Visualization / Results](#visualization--results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Motivation

Plant diseases cause significant losses in agriculture worldwide. Detecting diseases early—preferably through image-based automated solutions—can help farmers take timely action. This project supports such efforts by offering a lightweight, repeatable framework for leaf-disease classification.

## Features

- 🔍 **Preprocessing pipeline** with resizing, normalization, and augmentation
- 🧠 **Custom CNN** (or transfer learning) implementation
- 🧪 **Model training** with validation and checkpointing
- 📊 **Evaluation scripts** for accuracy, precision, recall, and F1-score
- 🖼️ **Visualization modules** for displaying sample predictions

## Getting Started

### Requirements

- Python 3.8+
- TensorFlow or PyTorch
- numpy, pandas, scikit-learn, matplotlib, opencv-python
- *(Optional)* CUDA + cuDNN for GPU acceleration

### Installation & Setup

### 🔧 Clone the Repository

```bash

git clone https://github.com/Shreya1sah/LeafDiseaseDetection.git
cd LeafDiseaseDetection/app
```

### 🐍 Create a Virtual Environment (optional)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 📦 Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 📁 Download or Prepare Data

Add your dataset in the following directory structure:

```bash
data/
  ├── train/
  │   ├── Apple_Scab/
  │   ├── Black_Rot/
  │   ├── Cedar_Apple_Rust/
  │   ├── Healthy/
  │   └── not_leaf/
  ├── test/
  │   ├── Apple_Scab/
  │   ├── Black_Rot/
  │   ├── Cedar_Apple_Rust/
  │   ├── Healthy/
  │   └── not_leaf/
  ├── val/
  │   ├── Apple_Scab/
  │   ├── Black_Rot/
  │   ├── Cedar_Apple_Rust/
  │   ├── Healthy/
  │   └── not_leaf/
```
You can use the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) or your own collection of labeled leaf images.
- The dataset used in this work consists of the Apple leaf disease dataset      from from PlantVillage dataset of Kaggle and not_leaf dataset.
- The not_leaf dataset contains the images of                                   trucks, cars, dogs, cats, humans, fruits, etc(images that are not leaves).
- The not_leaf dataset is used as if someone uploads the image of anything      that is not a leaf then the app correctly predicts that the image is          not_leaf.

### 🏋️‍♀️ Run Training

```bash
python train.py \
  --data_dir data/train \
  --epochs 20 \
  --batch_size 32 \
  --model output/leaf_model.h5
```

### 🧪 Test Model

```bash
python test.py \
  --data_dir data/test \
  --model output/leaf_model.h5 \
  --output_dir results/
```

## Project Structure

## Model & Evaluation

## Visualization / Results

## Contributing

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

## Acknowledgments
