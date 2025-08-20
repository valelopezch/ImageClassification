# ImageClassification

This repository contains a collection of projects focused on **image classification** using both **Keras** and **PyTorch**. Each project explores different datasets, neural network architectures, and training strategies, ranging from simple MLPs to advanced convolutional and transfer learning models.  

📄 **Note:** All detailed reports are written in **Spanish**. This README provides an English summary of the contents.  

---

## 📂 Projects Overview

### 1. Animals10 Classification (Keras)  
- **Dataset:** [Animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) (26k images, 10 animal classes).  
- **Methods:**  
  - Baseline with simple MLPs.  
  - CNN architectures with regularization and dropout.  
  - Transfer Learning with **EfficientNetB0** and **ResNet50**.  
  - Fine-tuning with data augmentation.  
- **Best Result:** 97.75% accuracy using EfficientNetB0 with fine-tuning + augmentation.  

---

### 2. CIFAR-10 Classification with Custom CNNs (PyTorch)  
- **Dataset:** CIFAR-10 (60k images, 10 classes).  
- **Methods:**  
  - Baseline CNN with 5 convolutional layers.  
  - Data augmentation (flips, rotations, shifts, crops).  
  - Residual blocks and learning rate scheduling.  
- **Best Result:** 92.64% accuracy using a deep residual CNN.  

---

### 3. CIFAR-10 with WideResNet and DenseNet (PyTorch)  
- **Dataset:** CIFAR-10.  
- **Architectures Tested:**  
  - **WideResNet:** depths {16, 22, 28}, widen factors {4, 6, 8}.  
  - **DenseNet:** multiple dense block configurations with growth rates.  
- **Best Results:**  
  - WideResNet-28-8: **94.1% accuracy** (23M parameters).  
  - DenseNet-BC3: **93.0% accuracy** with far fewer parameters.  

---

### 4. Gender Recognition with LFW (PyTorch)  
- **Dataset:** [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) (≈13k cropped face images).  
- **Methods:**  
  - Custom lightweight CNNs (ConvBlocks).  
  - Pretrained EfficientNet.  
  - Benchmarks: (i) >98% accuracy, (ii) >95% with <100k parameters.  
- **Best Results:**  
  - ConvBlock (32k params): **96.37% accuracy**.  
  - EfficientNet pretrained: **98.22% accuracy**.  

---

### 5. MNIST Digit Classification (PyTorch)  
- **Dataset:** MNIST (70k grayscale images of handwritten digits).  
- **Methods:**  
  - MLP baseline (no convolutions).  
  - Batch Normalization and learning rate annealing.  
  - Data augmentation (rotations, translations, scaling).  
- **Best Result:** **99.42% accuracy** using MLP + augmentation.  

---

## ⚙️ Tech Stack

- **Frameworks:**  
  - [PyTorch](https://pytorch.org/)  
  - [Keras](https://keras.io/) / [TensorFlow](https://www.tensorflow.org/)  

- **Core Libraries:**  
  - `torch`, `torchvision` – model building, training, datasets  
  - `tensorflow`, `keras` – deep learning models and transfer learning  
  - `numpy` – numerical operations  
  - `matplotlib`, `seaborn` – visualization and plots  
  - `scikit-learn` – evaluation metrics, classification reports  
  - `tqdm` – progress bars for training loops  

- **Development Environment:**  
  - [Jupyter Notebooks](https://jupyter.org/) for experimentation  
  - GPU acceleration via **Kaggle Notebooks** and local CUDA setups  

---

## 📑 Reports

All detailed results, methodologies, and conclusions are documented in **Spanish-language reports** located in the repository. These include:  

- `Presentacion_RFA.pdf` (Animals10, Keras CNNs + Transfer Learning)  
- `Informe_CIFAR_RNA.pdf` (CIFAR-10, CNNs + Residuals in PyTorch)  
- `Practicas_CIFAR10_CV.pdf` (CIFAR-10, WideResNet & DenseNet in PyTorch)  
- `Practicas_Gender_CV.pdf` (Gender recognition with LFW)  
- `Informe_MNIST_RNA.pdf` (MNIST classification with MLPs in PyTorch)  


## ✨ Key Takeaways

- Transfer Learning (EfficientNet, ResNet) significantly boosts performance on small and noisy datasets.

- Data augmentation and learning rate scheduling are crucial for generalization.

- WideResNet achieves state-of-the-art accuracy but at a high parameter cost, while DenseNet offers competitive results with fewer parameters.

- Even simple MLPs can achieve >99% accuracy on MNIST with the right augmentation strategy.

- Lightweight CNNs (<100k params) can achieve strong performance on real-world datasets like LFW.