# Customer Churn Prediction Model

A deep learning model to predict customer churn for a bank using TensorFlow and Keras.

## Overview

This project builds a neural network model to predict whether a bank customer will leave the bank (churn) or not. The model uses various customer attributes like credit score, geography, gender, age, tenure, balance, etc. to make predictions.

## Dataset

The model uses the Churn_Modelling.csv dataset which contains the following features:
- Customer demographic information (age, gender, geography)
- Banking relationship details (credit score, tenure, balance)
- Product usage (number of products, credit card status, active member status)
- Target variable: Exited (1 if customer left the bank, 0 otherwise)

Download the dataset from [here](https://github.com/YOUR-USERNAME/customer-churn-prediction/blob/main/Churn_Modelling.csv)

## Model Architecture

The neural network consists of:
- Input layer with 10 features
- First hidden layer with 10 neurons and ReLU activation
- Dropout layer (0.2) to prevent overfitting
- Second hidden layer with 16 neurons and ReLU activation
- Output layer with sigmoid activation for binary classification

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the Churn_Modelling.csv file from the GitHub repository and place it in your working directory.

5. Run the Jupyter notebook or Python script:
   ```
   jupyter notebook churn.ipynb
   # or
   python churn.py
   ```

## Results

The model achieves approximately 85% accuracy on the test set. The training and validation metrics show:
- Good balance between training and validation accuracy
- Stable loss reduction during training
- No significant signs of overfitting

## Future Improvements

- Feature engineering to create more predictive variables
- Hyperparameter tuning for improved performance
- Implementation of different algorithms for comparison
- Feature importance analysis for better interpretability

## License

This project is licensed under the MIT License - see the LICENSE file for details.
