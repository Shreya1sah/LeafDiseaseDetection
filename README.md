#                           🌿 SmartLeafAI

A fuzzy rank based ensemble learning pipeline has been used for detecting leaf diseases from images. Using preprocessing, model training, and visualization, the repository aims to provide an effortless workflow for disease classification in leaves.



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
- [Acknowledgements](#acknowledegments)

## Motivation

Leaf diseases cause significant losses in agriculture worldwide. Detecting diseases early—preferably through image-based automated solutions—can help farmers take timely action. This project supports such efforts by offering a lightweight, repeatable framework for leaf-disease classification.

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

### Download or Prepare Data

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
You can use the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) or your own collection of labeled leaf images.
- The dataset used in this work consists of the Apple leaf disease dataset from PlantVillage dataset of Kaggle and not_leaf dataset.
- The not_leaf dataset contains the images of                                   trucks, cars, dogs, cats, humans, fruits, etc(images that are not leaves).
- The not_leaf dataset is used as if someone uploads the image of anything      that is not a leaf then the app correctly predicts that the image is          not_leaf.

### Run Training

```bash
python train.py \
  --data_dir data/train \
  --epochs 20 \
  --batch_size 32 \
  --model output/leaf_model.h5
```

### Test Model

```bash
python test.py \
  --data_dir data/test \
  --model output/leaf_model.h5 \
  --output_dir results/
```
- The above structure generates performance metrics and sample prediction images in results/.
- leaf_model.h5 contains xception_model.h5, inceptionv3_model.h5, densenet169_model.h5 ,where xception,inceptionv3,densenet169 are the CNN models which are used to perform        fuzzy rank based ensemble approach for leaf disease detection

  
## Project Structure

```plaintext
LeafDiseaseDetection/
├── app/                          # Android application code
│   ├── java/                     # Java source files for app logic
│   ├── assets/                   # TensorFlow Lite models (.tflite files)
│   └── res/                      # UI layouts, icons, and other resources
├── models/                      
├── scripts/                      # Python scripts for training and fusion logic
│   ├── fuzzy_ensemble.py         # Fuzzy rank-based ensemble implementation
├── data/                         # Image dataset organized for training/testing
│   ├── train/
│   ├── test/
│   └── val/
├── requirements.txt              # Python dependencies (for model development)
└── README.md                     # Project documentation
```

"data/" section has been defined above in "Download or Prepare Data" section

## Model & Evaluation

This project uses an ensemble of three deep learning models for leaf disease classification:

- 🧠 **DenseNet169**
- 🧠 **InceptionV3**
- 🧠 **Xception**

Each model is trained independently on the same dataset. Their softmax outputs are passed into a **Fuzzy Rank-Based Ensemble** module which performs the following:

- 🔄 **Normalization** of confidence scores into fuzzy ranks  
- ➕ **Aggregation** of fuzzy ranks across all models  
- ✅ **Final decision** based on the minimum total rank sum (i.e., highest confidence)

This fuzzy ensemble method increases robustness, especially in differentiating between diseased leaves and completely unrelated inputs such as:

- 🚗 trucks
- 🐶 dogs
- 👨 humans
- 🍎 fruits

These are captured under a special class: **`not_leaf`**, to avoid false predictions when the input is not a leaf.

---

### Evaluation Metrics

The model is evaluated on the test set using the following metrics:

- 📈 Accuracy  
- 🎯 Precision  
- ♻️ Recall  
- 📊 F1-Score  
- 🔁 Confusion Matrix

All results, including numeric metrics and visual outputs (like confusion matrices and prediction examples), needs to be saved in the `results/` directory.


## Visualization / Results

The `results/` directory should include:

- 📉 **Confusion matrix plots**
- 🖼️ **Sample predictions** (images with predicted labels)
- 📊 **Per-class accuracy scores**
- 📈 **Comparison charts** of individual model performance vs. ensemble

---

These visualizations help verify the model’s effectiveness:

- ✅ **Healthy vs. diseased leaves** are correctly classified
- 🚫 **Non-leaf images** (e.g., dogs, cars, fruits) are not misclassified as diseased
- 🤝 The **fuzzy rank-based ensemble** improves reliability by reducing false positives and negatives, especially for ambiguous or noisy samples

## Contributing

Contributions, issues, and feature requests are welcome!  
If you’d like to contribute:

1. 🍴 Fork the repository
2. 📥 Clone your fork
3. 🛠️ Create a new branch for your feature or bugfix
4. 💡 Make your changes and commit them
5. 🔄 Push to your branch
6. 📩 Open a pull request

Please ensure your code follows best practices and includes relevant comments and documentation where applicable.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

## Acknowledgements

- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — for providing a diverse dataset of leaf disease images.
- TensorFlow and Keras teams — for enabling powerful deep learning workflows.
- The open-source community — for providing incredible tools, resources, and support.
- All researchers and contributors in the field of leaf disease detection.

Special thanks to everyone who helped test and improve this application.

