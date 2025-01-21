# Analiza si Compararea Metoddelor Supervizate de Clasificare pe Imagini Hiperspectrale

## Overview
This project analyzes and compares supervised classification methods applied to hyperspectral image datasets. Techniques like Support Vector Machines (SVM), Neural Networks (NN), and Random Forest are implemented and evaluated on a dataset of hyperspectral images. The goal is to assess the performance of these algorithms and provide insights into their advantages and disadvantages.

## Dataset
The dataset used in this project is available on Kaggle: [Apple Hyperspectral Images Dataset](https://www.kaggle.com/datasets/warcoder/apple-hyperspectral-images-dataset). Preprocessing steps include resizing images for consistency and normalizing data for SVM. Dimensionality reduction using PCA is also performed.

## Features
1. **Recursive Image Loading:** Images are loaded from folders and subfolders, resized for consistency.
2. **Dimensionality Reduction:** PCA is applied to reduce data complexity.
3. **Algorithm Implementation:**
   - **Support Vector Machines (SVM):** Training and testing phases with performance evaluation.
   - **Neural Networks (NN):** Implementation and evaluation of deep learning methods.
   - **Random Forest:** Performance evaluation and comparison with other methods.
4. **Performance Metrics:** Accuracy and other metrics are computed for each method. Graphs are plotted to visualize performance.

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `rasterio`
- `skimage`
- `os`, `sys`

## Results
- **Random Forest:** Achieved the best results due to its ensemble approach, reducing overfitting and handling noisy data well.
- **SVM:** Suitable for smaller datasets; less prone to overfitting but slower for large datasets.
- **NN:** Highly flexible but sensitive to hyperparameter selection and prone to overfitting without sufficient data.

## Advantages and Disadvantages of Methods
### SVM:
- **Advantages:** Effective on small datasets, kernel flexibility.
- **Disadvantages:** Slow for large datasets, challenging hyperparameter tuning.

### Random Forest:
- **Advantages:** Combines decision trees for better generalization.
- **Disadvantages:** High memory usage, slower compared to SVM and NN.

### Neural Networks:
- **Advantages:** Can be extended and adjusted with layers and neurons.
- **Disadvantages:** Overfitting risk with small datasets, sensitive to learning rate and activation function.

## Getting Started
1. Clone this repository.
2. Install the required libraries using the following command:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn rasterio scikit-image
   ```
3. Download the dataset from Kaggle and place it in the appropriate directory.
4. Run the notebook step-by-step to preprocess data, train models, and evaluate their performance.

## Conclusion
This project highlights the strengths and weaknesses of different supervised classification methods when applied to hyperspectral images. Random Forest proved to be the most robust approach, while SVM and NN offer distinct advantages for specific scenarios.

---
Feel free to contribute to this project by suggesting improvements or adding new algorithms!

