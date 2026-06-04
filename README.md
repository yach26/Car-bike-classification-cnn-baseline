# Car vs. Bike Classification: CNN Baseline Model

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

A baseline Convolutional Neural Network (CNN) built in TensorFlow and Keras to classify images into two categories: **Cars** and **Bikes**. The model uses a classic sequential architecture with image preprocessing, data augmentation, convolutional feature extraction, pooling, dropout regularization, and binary classification.

---

## 📂 Project Structure

```
Model_2_CNN/
│
├── cnn-car-vs-bike.ipynb                         # Jupyter Notebook containing the baseline CNN code
├── cnn-car-vs-bike-with-a-pretrained-model.ipynb     # Notebook containing baseline CNN & MobileNetV2 models
├── README.md                                     # Detailed project documentation and deep learning theory
├── .gitignore                                    # Ignores python cache, checkpoints, and IDE folders
└── mobilenetv2_car_bike/
    └── mobilenetv2_car_bike.keras                # Saved Keras model file using MobileNetV2
```

---

## 📊 Dataset Overview

The model is trained on the [Car vs Bike Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset) from Kaggle.
* **Total Images**: 4,000 images
  * **Car Images**: 2,000
  * **Bike Images**: 2,000
* **Class Balance**: 1:1 (perfectly balanced dataset)
* **Data Split**: 80% for training (3,200 images), 20% for validation (800 images).

---

## 🔄 Data Pipeline & Augmentation

Deep neural networks require large amounts of data to generalise well. To prevent overfitting and help the model learn spatial-invariance, data augmentation is performed using Keras's `ImageDataGenerator` with the following parameters:

```python
datagen = ImageDataGenerator(
    rescale=1./255,           # Normalises pixel values from [0, 255] to [0.0, 1.0]
    validation_split=0.2,     # Reserves 20% of dataset for validation
    rotation_range=20,        # Randomly rotates images up to 20 degrees
    zoom_range=0.2,           # Randomly zooms inside pictures by 20%
    horizontal_flip=True      # Randomly flips half of the images horizontally
)
```

During ingestion:
* All images are resized to a target size of **$224 \times 224$ pixels** with **3 channels (RGB)**.
* Images are loaded in batches of **32** during training.

---

## 🧠 Model Architecture & Deep Learning Theory

The network follows a **Sequential** layout starting with an input layer, passing through three stages of Convolution + MaxPooling, flattening, passing through a Dense layer with Dropout, and concluding with a single-neuron Sigmoid output.

```
Input (224x224x3) ──> Conv2D(32) ──> MaxPool2D ──> Conv2D(64) ──> MaxPool2D 
                  ──> Conv2D(128) ──> MaxPool2D ──> Flatten ──> Dense(128) 
                  ──> Dropout(0.5) ──> Dense(1, Sigmoid) ──> Binary Output
```

### 1. Convolutional Layer (`Conv2D`)
A convolutional layer extracts spatial features (such as edges, corners, textures, and shapes) by sliding small Learnable filters (kernels) over the input image.

* **Mathematical Operation**:
  Each filter performs an element-wise multiplication and summation (dot product) over a local receptive field:
  $$S(i,j) = (I * K)(i,j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m,n)$$
  Where $I$ is the input matrix, $K$ is the kernel matrix, and $S$ is the output feature map.
* **Activation Function (ReLU)**:
  Every Convolutional layer is followed by a **Rectified Linear Unit (ReLU)** activation:
  $$f(x) = \max(0, x)$$
  ReLU introduces non-linearity to the network, enabling it to learn complex non-linear relationships. It also helps alleviate the *vanishing gradient problem* because its derivative is $1$ for all positive inputs.

### 2. Max Pooling (`MaxPooling2D`)
Pooling layers downsample the feature maps to reduce spatial dimensions, reducing computational complexity and the number of parameters.

* **Operation**:
  A window of size $2 \times 2$ slides across the feature map and retains only the maximum value in that window.
* **Why it is used**:
  * **Translation Invariance**: Helps the model detect features regardless of their exact position in the image.
  * **Feature Selection**: Keeps the most prominent activations and discards weaker ones.

### 3. Flattening (`Flatten`)
* **Operation**:
  Transforms the multi-dimensional output tensor of the last pooling layer (a 3D array of shape `(batch_size, height, width, channels)`) into a 1D vector (shape `(batch_size, height * width * channels)`).
* **Purpose**:
  Prepares the multidimensional spatial feature maps for the fully connected neural network layers.

### 4. Dense (Fully Connected) Layer (`Dense`)
* **Operation**:
  Every input neuron is connected to every output neuron in this layer. It computes the dot product of inputs and weights, adds a bias, and applies an activation function:
  $$y = f(W \cdot x + b)$$
  This layer aggregates all the localized features extracted by the convolutional layers to perform the classification decision.

### 5. Dropout Layer (`Dropout`)
* **Operation**:
  During each training epoch, a random fraction (here, $50\%$) of the input neurons to the next layer are set to zero.
* **Why it is used**:
  Dropout is a powerful regularization technique. It forces the network to learn redundant representations and prevents co-adaptation of neurons, which heavily reduces **overfitting**.

### 6. Output Activation (`Sigmoid`)
For binary classification, the final output layer consists of a single unit with a **Sigmoid** activation function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* The output value represents the probability that the input image belongs to class 1 (Car):
  $$P(\text{Class} = \text{Car} \mid X) = \sigma(z)$$
  $$P(\text{Class} = \text{Bike} \mid X) = 1 - \sigma(z)$$
* A threshold of $0.5$ is applied to determine the final class.

---

## ⚙️ Optimization & Loss Function

### 1. Loss Function: Binary Cross-Entropy
Since this is a binary classification task, the model is compiled with the **Binary Cross-Entropy (Log Loss)** function:
$$\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$
Where:
* $N$ is the number of samples in the batch.
* $y_i$ is the actual ground truth label ($0$ for Bike, $1$ for Car).
* $\hat{y}_i$ is the predicted probability of the image being a Car.

### 2. Optimizer: Adam
The model parameters are optimized using the **Adam (Adaptive Moment Estimation)** optimizer.
* **How it works**:
  Adam computes adaptive learning rates for each parameter. It stores both an exponentially decaying average of past gradients (first moment, like Momentum) and past squared gradients (second moment, like RMSProp):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  Where $g_t$ is the gradient, and $\beta_1, \beta_2$ are hyperparameters (defaulting to $0.9$ and $0.999$). This allows for stable convergence and faster learning.

---

## 📈 Architecture Summary

Below is the summary of the network's layers, output shapes, and parameter counts:

| Layer (type) | Output Shape | Param # | Description |
| :--- | :--- | :--- | :--- |
| **Input** | `(None, 224, 224, 3)` | 0 | Input image dimensions |
| **Conv2D (1st)** | `(None, 222, 222, 32)` | 896 | 32 filters of size $3\times3$ |
| **MaxPooling2D** | `(None, 111, 111, 32)` | 0 | Spatial resolution halved |
| **Conv2D (2nd)** | `(None, 109, 109, 64)` | 18,496 | 64 filters of size $3\times3$ |
| **MaxPooling2D** | `(None, 54, 54, 64)` | 0 | Spatial resolution halved |
| **Conv2D (3rd)** | `(None, 52, 52, 128)` | 73,856 | 128 filters of size $3\times3$ |
| **MaxPooling2D** | `(None, 26, 26, 128)` | 0 | Spatial resolution halved |
| **Flatten** | `(None, 86528)` | 0 | Flattened 1D vector ($26 \times 26 \times 128$) |
| **Dense** | `(None, 128)` | 11,075,712 | Fully connected representation |
| **Dropout** | `(None, 128)` | 0 | 50% neuron dropout for regularization |
| **Dense (Output)** | `(None, 1)` | 129 | 1 sigmoid output unit |

* **Total trainable parameters**: 11,169,089 (~11.17 Million)

---

## 🧬 Transfer Learning & Pretrained Models

In the second version of the model, we transition from our custom baseline CNN to **Transfer Learning** using a state-of-the-art pretrained architecture: **MobileNetV2**.

### 1. What is a Pretrained Model?
A **pretrained model** is a deep learning model that has been previously trained on a massive, benchmark dataset (such as **ImageNet**, which contains over 14 million images categorized into 1,000 diverse classes). 
Instead of training a neural network from scratch with randomized weights, we load this model with its pre-learned weights. The early and middle layers of these models have already learned general feature extractors (e.g., edges, textures, shapes, spatial patterns) that are highly transferable to other computer vision tasks.

### 2. Why Use Transfer Learning? (Benefits & Usefulness)
* **High Accuracy with Limited Data**: Training deep CNNs from scratch on small datasets (like our 4,000-image dataset) typically leads to severe overfitting. Pretrained models bring prior knowledge, allowing them to generalize exceptionally well even with limited samples.
* **Dramatic Computational Savings**: Instead of training millions of parameters for days, we freeze the pretrained weights and only train a small "classifier head" on top. This reduces the number of trainable parameters and speeds up training time.
* **Faster Convergence**: Pretrained models start with optimized feature detectors, allowing the training loss to decrease much faster and stabilize in just a few epochs (e.g., 5 epochs instead of 10+).

### 3. How Transfer Learning Works: Freezing vs. Fine-Tuning
Transfer learning generally follows two steps:
1. **Feature Extraction (Freezing)**: 
   We load the base network (e.g., MobileNetV2) without its final classification layer (`include_top=False`). We freeze all its layers (`model.trainable = False`), meaning their weights will not be updated during backpropagation. We add a custom classification head on top and train only this head.
2. **Fine-Tuning (Optional)**:
   After the custom head is trained, we can unfreeze a few of the top layers of the base network and train the model again with a very low learning rate. This adapts the high-level features of the pretrained base to our specific dataset.

In this project, we implement **Feature Extraction** by freezing the entire MobileNetV2 base.

---

### 📱 Pretrained Architecture: MobileNetV2
**MobileNetV2** is a highly efficient convolutional neural network architecture developed by Google, optimized for mobile and resource-constrained devices. It features:
* **Depthwise Separable Convolutions**: Splits standard convolution into:
  1. *Depthwise Convolution*: A single spatial filter applied per input channel.
  2. *Pointwise Convolution*: A $1\times1$ convolution that mixes the channels.
  This reduces the computational cost and parameter count by a factor of 8 to 9 compared to standard convolutions with only a tiny drop in accuracy.
* **Inverted Residual Blocks**: Traditional residual blocks connect layers with many channels. MobileNetV2 connects thin bottleneck layers instead, expanding them temporarily to extract features and then projecting them back to prevent information loss.
* **Linear Bottlenecks**: Prevents non-linearities (like ReLU) from destroying useful information in low-dimensional spaces.

#### Pretrained Model Pipeline (Keras Sequential)
```
[Input (224x224x3)] 
        │
[MobileNetV2 Base (Weights: ImageNet, Trainable: False)] (Output: 7x7x1280)
        │
[GlobalAveragePooling2D] (Reduces spatial dimensions to a 1D vector of 1280 features)
        │
[Dense (128, ReLU)] (Intermediate fully connected layer)
        │
[Dropout (0.3)] (Regularization layer preventing overfitting)
        │
[Dense (1, Sigmoid)] ──> [Binary Output (Bike vs. Car)]
```

---

## 📈 Model Comparison & Metrics

We compared our custom **Baseline CNN** against the **MobileNetV2 Pretrained Model** on the validation set of 800 images:

| Metric | Baseline CNN (10 Epochs) | MobileNetV2 Pretrained (5 Epochs) |
| :--- | :---: | :---: |
| **Total Trainable Params** | 11,169,089 (~11.17M) | 164,097 (~0.16M) |
| **Training Time (per epoch)** | ~220 seconds | **~120 seconds** (45% faster) |
| **Training Accuracy** | ~92.09% | **~98.69%** |
| **Validation Accuracy** | ~90.00% | **~98.75%** |
| **False Positives (Predicted Car for Bike)**| 54 | **7** |
| **False Negatives (Predicted Bike for Car)**| 22 | **3** |

### Confusion Matrix Comparison

```
   Baseline CNN Matrix                  MobileNetV2 Matrix
   
      Predicted                            Predicted
     Bike   Car                           Bike   Car
Actual                               Actual
 Bike [346   54]                      Bike  [393    7]
 Car  [ 22  378]                      Car   [  3  397]
```

### Classification Report (MobileNetV2)
```
              precision    recall  f1-score   support

        Bike       0.99      0.98      0.99       400
         Car       0.98      0.99      0.99       400

    accuracy                           0.99       800
   macro avg       0.99      0.99      0.99       800
weighted avg       0.99      0.99      0.99       800
```

---

## 🚀 How to Run the Notebooks

### Prerequisites
Make sure you have Python 3 installed along with the required libraries:
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn pillow
```

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yach26/Car-bike-classification-cnn-baseline.git
   cd Car-bike-classification-cnn-baseline
   ```
2. Download the dataset and place it in the path specified inside the notebooks.
3. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. Choose the notebook you wish to run:
   * **Baseline Model**: Open and run `cnn-car-vs-bike.ipynb`.
   * **Transfer Learning Model**: Open and run `cnn-car-vs-bike-with-a-pretrained-model.ipynb`.

---

## 📊 Evaluation & Metrics

Both notebooks evaluate model performance on validation data using the following metrics:
1. **Accuracy Score**: Overall percentage of correct predictions.
2. **Confusion Matrix**: Identifies True Positives, True Negatives, False Positives, and False Negatives.
3. **Classification Report**:
   * **Precision**: Out of all predicted Cars/Bikes, how many were actually Cars/Bikes?
   * **Recall**: Out of all actual Cars/Bikes, how many did the model find?
   * **F1-Score**: Harmonic mean of Precision and Recall.

Training curves for validation and training accuracy/loss are plotted at the end of each notebook to monitor convergence and check for overfitting.
