# Car Make and Model Recognition System

## Overview

This project implements a deep learning solution to automatically recognize car makes and models from images. Using a pre-trained ResNet-34 model fine-tuned on a custom dataset (sourced from the Stanford Car Dataset and Kaggle), the system is capable of classifying 196 unique car classes with high accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset and Data Requirements](#dataset-and-data-requirements)
- [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Model Architecture and Training](#model-architecture-and-training)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Evaluation](#evaluation)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Future Extensions](#future-extensions)
- [License](#license)

## Introduction

The Car Make and Model Recognition System is a university project aimed at developing an image classification model that distinguishes between different car models based on their make, model, and year. The project includes data collection, pre-processing (including data augmentation and cleaning), model training with hyperparameter tuning, and evaluation using standard metrics such as accuracy, precision, recall, and F1-score.

## Dataset and Data Requirements

### Data Types

- **Numerical Data:**
  - **Vehicle Year** (e.g., 2012, 2020)
  - *Purpose:* Distinguishing between different model years.

- **Categorical Data:**
  - **Car Make and Model** (e.g., BMW M3, Tesla Model S)
  - *Purpose:* Core classification labels for the model.

- **Image Data:**
  - **Car Images:**
    - *Requirements:* Clear images with a resolution of at least 400x400 pixels.
    - *Sources:* Stanford Car Dataset, Kaggle, automotive databases.
    - *Ethical Note:* All data is publicly available or legally acquired for academic work.

### Data Organization

The dataset is structured into two main directories:

```bash
car_data/
├── train/
│   ├── [Car_Class_1]/
│   ├── [Car_Class_2]/
│   └── ...
└── test/
    ├── [Car_Class_1]/
    ├── [Car_Class_2]/
    └── ...
```

Each subdirectory is named using a "Year Make Model" format (e.g., `Hyundai Accent Sedan 2012`).

## Data Preprocessing and Augmentation

### Blurry Image Detection

To improve dataset quality, images are assessed for blur using the Laplacian variance method via OpenCV. Blurry images are then moved to a separate folder:

```python
def is_blurry(image_path, threshold=100):
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var() < threshold
```

### Data Augmentation

Underrepresented classes (e.g., *Hyundai Accent Sedan 2012*) are augmented with transformations such as:
- **Random Horizontal Flip**
- **Random Rotation (±15°)**
- **Color Jitter**

Example augmentation function:

```python
def augment_images(image_folder, class_name, target_count):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    ])
    class_path = os.path.join(image_folder, class_name)
    images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
    num_images = len(images)
    while num_images < target_count:
        img_path = random.choice(images)
        img = Image.open(img_path)
        new_img = transform(img)
        new_img_path = os.path.join(class_path, f"aug_{num_images}.jpg")
        new_img.save(new_img_path)
        num_images += 1
```

## Model Architecture and Training

### Model Setup

A pre-trained ResNet-34 model is fine-tuned for the classification task. The final fully connected layer is modified to output predictions for 196 classes:

```python
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)
```

### Training Process

- **Loss Function:** `nn.CrossEntropyLoss()`
- **Optimizer:** `optim.SGD` (with parameters optimized via Optuna)
- **Learning Rate Scheduler:** `StepLR`

## Hyperparameter Optimization

Optuna is used to optimize key hyperparameters such as:
- **Learning Rate**
- **Momentum**
- **Step Size** (for the scheduler)
- **Gamma** (learning rate decay factor)

## Evaluation

After training, the model is evaluated on a test set using the following metrics:
- **Loss**
- **Accuracy**
- **Precision, Recall, and F1-score**

## Installation and Setup

### Requirements

- Python 3.x
- PyTorch
- TorchVision
- OpenCV
- Optuna
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/car-make-model-recognition.git
   cd car-make-model-recognition
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Training Script

To train the model, run:

```bash
python train.py
```

### Evaluating the Model

To evaluate the trained model, execute:

```bash
python evaluate.py --model_path model_resnet34.pth
```

## Future Extensions

- **Experimenting with Other Architectures:** Evaluate models like ResNet-50, EfficientNet, or Vision Transformers.
- **Model Explainability:** Implement Grad-CAM or SHAP to visualize model decision regions.
- **Deployment:** Build a REST API using Flask or FastAPI to serve the model for real-time inference.
- **Handling Imbalanced Data:** Implement oversampling methods or class-weighted loss functions for better performance on underrepresented classes.


