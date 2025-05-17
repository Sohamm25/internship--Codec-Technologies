import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = pd.DataFrame({'movie_title': ['Movie A', 'Movie B', 'Movie C'],
                     'other_column': [1, 2, 3]})
similarity = np.array([[1.0, 0.8, 0.6],
                       [0.8, 1.0, 0.7],
                       [0.6, 0.7, 1.0]])

try:
    print("Head of the data:")
    print(data.head())  # Access the head of the data DataFrame
    print("\nShape of the similarity matrix:")
    print(similarity.shape)  # Access the shape of the similarity matrix
    print("\nData access succeeded!")
except:
    print("Data access failed!")
    # If access fails, create similarity scores
    data, similarity = create_similarity()
