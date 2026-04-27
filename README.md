# Automated crack detection

An academic/engineering project focused on automated crack detection in industrial infrastructure (e.g., concrete surfaces) using deep learning.

## About the project
To solve the binary classification problem (crack vs. no crack), a Convolutional Neural Network (CNN) was built and trained using the **TensorFlow/Keras** library.

The dataset was split into three independent parts:
* **Training set (80%)** - used to train the model.
* **Validation set (10%)** - used to monitor the training process and prevent overfitting.
* **Test set (10%)** - used for the final, objective evaluation of the model's performance on completely unseen data.

## Technologies
* Python
* TensorFlow / Keras
* Matplotlib & Seaborn (Data Visualization)
* Scikit-learn (Evaluation Metrics)

## Model results
The model achieved high accuracy on the test set. Below are the visualizations of the training process and the final confusion matrix.

### Training history (accuracy and loss)
![Training History](wykres_uczenia.png)

### Confusion matrix
![Confusion Matrix](macierz_pomylek.png)

## How to Run the Project locally
1. Clone this repository.
2. Download the dataset and place the images inside the `dataset/` folder (using `positive` and `negative` subfolders).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt