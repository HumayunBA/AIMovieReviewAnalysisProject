import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Data Loading
def load_dataset(folder_path):
    reviews = []
    labels = []
    for label in ['pos', 'neg']:
        label_folder = os.path.join(folder_path, label)
        for file in os.listdir(label_folder):
            with open(os.path.join(label_folder, file), 'r', encoding='utf-8') as f:
                review = f.read()
            reviews.append(review)
            labels.append(label)
    return reviews, labels



# Step 3: Feature Extraction
def extract_features(reviews):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)
    return X

# Step 4: Model Training
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function
def main():
    # Load dataset
    train_reviews, train_labels = load_dataset(r'C:\Users\Humayun\Documents\aclImdb\train')
    test_reviews, test_labels = load_dataset(r'C:\Users\Humayun\Documents\aclImdb\test')

    # Feature extraction
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_reviews)  
    X_test = vectorizer.transform(test_reviews)      

    # Model training
    model = train_model(X_train, train_labels)

    # Model evaluation
    accuracy = evaluate_model(model, X_test, test_labels)
    print("Accuracy:", accuracy)
    
    # Visualize the distribution of labels in the training set
    sns.countplot(train_labels)
    plt.title('Distribution of Sentiment Labels in Training Set')
    plt.show()
    
    # Visualize the distribution of labels in the test set
    sns.countplot(test_labels)
    plt.title('Distribution of Sentiment Labels in Test Set')
    plt.show()

if __name__ == "__main__":
    main()
