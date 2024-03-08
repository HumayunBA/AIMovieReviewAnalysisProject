# AIMovieReviewAnalysisProject

Problem Description:

I will focus on analyzing movie reviews to determine their sentiment, aiming to classify them as positive or negative. By using machine learning techniques, I aim to develop a model capable of accurately identifying the sentiment expressed in these reviews.

Data Collection and Analysis:

I collected the dataset of movie reviews along with their sentiment labels (positive or negative) primarily from the Stanford university AI project which is available for public.

Using Python libraries such as pandas and numpy, I analyzed the dataset to understand its structure, quality, and any potential issues. Additionally, I preprocessed the data to handle missing values and converted the text into a suitable format for analysis.

During the analysis, it was observed that the dataset contained an equal number of positive and negative movie reviews, as illustrated in the distribution of sentiment labels chart. This balanced distribution is crucial for training machine learning models to ensure unbiased performance evaluation.

Furthermore, the text data was preprocessed to handle missing values and converted into a suitable format for analysis using techniques like tokenization, stop words removal, and TF-IDF vectorization.

Type of Problem:

This is a classification problem where I aim to classify movie reviews as either positive or negative based on their sentiment. To train the model effectively, labeled data where each review is associated with its corresponding sentiment label is required.

Future Considerations:

For the sentiment analysis task, I am considering the use of deep learning techniques such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs). I will explore various architectures and fine-tuning methods to improve the model's performance.

Furthermore, I will investigate different training platforms and resources such as TensorFlow and PyTorch to ensure efficient model training and deployment.

Documentation:

The project in the first step focused on the problem description, data collection and analysis, type of problem, and future considerations. In the second step, the implementation was done in Jupyter Notebook and Python, please refer to the uploaded code repository.

Following libraries have been used:

os
matplotlib.pyplot as plt
seaborn as sns
pandas as pd
sklearn.model_selection.train_test_split
sklearn.feature_extraction.text.TfidfVectorizer
sklearn.linear_model.LogisticRegression
sklearn.metrics.accuracy_score
sklearn.metrics.classification_report
sklearn.metrics.confusion_matrix



Reference for dataset: 

  Link: https://ai.stanford.edu/~amaas/data/sentiment/

  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}


