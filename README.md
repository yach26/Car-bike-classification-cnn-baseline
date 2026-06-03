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
├── cnn-car-vs-bike.ipynb   # Jupyter Notebook containing the training and evaluation code
├── README.md               # Detailed project documentation and deep learning theory
└── .gitignore              # Ignores python cache, checkpoints, and IDE folders
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

## 🚀 How to Run the Notebook

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
2. Download the dataset and place it in the path specified inside the notebook.
3. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. Open `cnn-car-vs-bike.ipynb` and run all cells.

---

## 📊 Evaluation & Metrics

The notebook evaluates model performance on validation data using the following metrics:
1. **Accuracy Score**: Overall percentage of correct predictions.
2. **Confusion Matrix**: Identifies True Positives, True Negatives, False Positives, and False Negatives.
3. **Classification Report**:
   * **Precision**: Out of all predicted Cars/Bikes, how many were actually Cars/Bikes?
   * **Recall**: Out of all actual Cars/Bikes, how many did the model find?
   * **F1-Score**: Harmonic mean of Precision and Recall.

Training curves for validation and training accuracy/loss are plotted at the end of the notebook to monitor convergence and check for overfitting.
