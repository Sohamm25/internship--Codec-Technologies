# Data Science and Machine Learning Projects

This repository contains two complete projects:
1. Movie Recommendation System
2. Customer Churn Prediction Model 
 
## SCREENSHOTS OF PROJECT-
https://drive.google.com/drive/folders/18ZoUBO9Qi_Sz1GDYyhaE8MeGnTfBmS3B?usp=sharing
 
## 1. Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on user preferences, built with Flask, scikit-learn, and NLP techniques.

### Features

- Movie search with autocomplete functionality
- Content-based movie recommendations
- Display of movie details including cast, overview, and ratings
- Real-time IMDb user reviews with sentiment analysis
- Responsive UI for better user experience

### Technologies Used

- Python 3.x
- Flask
- scikit-learn
- NumPy & Pandas
- BeautifulSoup4 for web scraping
- TMDB API for movie data
- Natural Language Processing for sentiment analysis
- HTML, CSS, JavaScript & Bootstrap

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Sohamm25/data-science-projects.git
   cd data-science-projects/movie-recommendation-system
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

4. Download the required data files and place them in the project directory:
   - [Download NLP model & transform files](https://drive.google.com/file/d/1NoqW0d70rymoBcqShO-v-lFxwmMmSGyL/view?usp=drive_link)
   - [Download dataset files](https://drive.google.com/file/d/1CEBvXUE7_Sj3JGF5lrDMWc4j4mETdXxR/view?usp=sharing)
   
   Extract these files and place them in a `datasets` folder within your project directory.

5. **How to get the API key?**
   Create an account in https://www.themoviedb.org/, click on the `API` link from the left hand sidebar in your account settings and fill all the details to apply for API key. If you are asked for the website URL, just give "NA" if you don't have one. You will see the API key in your `API` sidebar once your request is approved.

6. Run the application:
   ```
   python main.py
   ```

7. Open your browser and navigate to `http://localhost:5000/`

### How It Works

1. **Content-Based Filtering**: The system analyzes movie metadata (genres, keywords, cast, crew) to find similarities
2. **Cosine Similarity**: Measures the similarity between movies based on their feature vectors
3. **Sentiment Analysis**: Analyzes IMDb reviews to determine positive/negative sentiment

### Project Structure

```
movie-recommendation-system/
├── main.py                  # Main application file
├── static/                  # Static files (CSS, JS, images)
├── templates/               # HTML templates
├── datasets/                # Data files
│   ├── main_data.csv        # Movie dataset
│   ├── nlp_model.pkl        # Trained NLP model
│   └── tranform.pkl         # TF-IDF vectorizer
└── requirements.txt         # Required packages
```

## 2. Customer Churn Prediction Model

A deep learning model to predict customer churn for a bank using TensorFlow and Keras.

### Overview

This project builds a neural network model to predict whether a bank customer will leave the bank (churn) or not. The model uses various customer attributes like credit score, geography, gender, age, tenure, balance, etc. to make predictions.

### Dataset

The model uses the Churn_Modelling.csv dataset which contains the following features:
- Customer demographic information (age, gender, geography)
- Banking relationship details (credit score, tenure, balance)
- Product usage (number of products, credit card status, active member status)
- Target variable: Exited (1 if customer left the bank, 0 otherwise)

Download the dataset from [here](https://github.com/YOUR-USERNAME/data-science-projects/blob/main/customer-churn-prediction/Churn_Modelling.csv)

### Model Architecture

The neural network consists of:
- Input layer with 10 features
- First hidden layer with 10 neurons and ReLU activation
- Dropout layer (0.2) to prevent overfitting
- Second hidden layer with 16 neurons and ReLU activation
- Output layer with sigmoid activation for binary classification

### Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Installation

1. Navigate to the churn prediction directory:
   ```
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

### Results

The model achieves approximately 85% accuracy on the test set. The training and validation metrics show:
- Good balance between training and validation accuracy
- Stable loss reduction during training
- No significant signs of overfitting

## Future Improvements

### Movie Recommendation System
- Add user authentication system
- Implement collaborative filtering
- Add more data sources for richer recommendations
- Create a watchlist feature

### Customer Churn Prediction
- Feature engineering to create more predictive variables
- Hyperparameter tuning for improved performance
- Implementation of different algorithms for comparison
- Feature importance analysis for better interpretability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- TMDB API for providing movie data
- IMDb for review data
