from argparse import ArgumentParser
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.util import ngrams

parser = ArgumentParser()

parser.add_argument("--train", type=str, required=True)
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--stop", type=str, required=True)

args = parser.parse_args()

train_data_path = args.train
test_data_path = args.test
output_path = args.out
stop_path = args.stop

train_df= pd.read_csv(train_data_path,header=None,sep='\t',quoting=3)
train_x= train_df[2]
train_y= train_df[1]
stop= pd.read_csv(stop_path,header=None)[0]
stop=stop.astype(str).tolist()

stemmer = PorterStemmer()

def preprocess_text(text):
    words = text.lower().split()
    # Filter out stopwords and apply stemming
    filtered_words = [stemmer.stem(word) for word in words if word not in stop]

    # Create unigrams (individual words)
    unigrams = filtered_words
    
    # Create bigrams (pairs of consecutive words)
    bigrams = ['_'.join(bigram) for bigram in ngrams(filtered_words, 2)]
    
    # Combine unigrams and bigrams
    return unigrams + bigrams

train_x= train_x.apply(preprocess_text)

label_to_index = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true': 5}
index_to_label = {v: k for k, v in label_to_index.items()}

def create_feature_matrix(processed_texts, labels):
    vocabulary = set(word for text in processed_texts for word in text)
    vocab_size = len(vocabulary)
    
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    
    feature_matrix = np.zeros((len(processed_texts), vocab_size), dtype=int)
    class_count = np.zeros(len(set(labels)))

    # Fill the feature matrix
    for i, text in enumerate(processed_texts):
        for word in text:
            if word in word_to_index:
                feature_matrix[i, word_to_index[word]] += 1 

    # Encode labels
    encoded_labels = np.array([label_to_index[label] for label in labels])
    
    # Calculate class count as the sum of words in sentences of each class
    for i, label in enumerate(encoded_labels):
        class_count[label] += feature_matrix[i].sum()  # Sum of words in the sentence

    return feature_matrix, encoded_labels, class_count, word_to_index


X, Y, class_count, word_to_index= create_feature_matrix(train_x, train_y)

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.feature_probs = None
        self.unseen_word_value_per_class = None

    def fit(self, X, y, class_count):
        self.class_priors = np.log(np.bincount(y) / len(y))
        self.feature_probs = np.zeros((len(self.class_priors), X.shape[1]))
        for c in range(len(self.class_priors)):
            X_c = X[y == c]
            self.feature_probs[c] = (X_c.sum(axis=0) + 1) / (X_c.sum() + X.shape[1])
        
        # class_count = np.bincount(y)
        self.unseen_word_value_per_class = [1 / (X.shape[1] + count) for count in class_count]

    def predict(self, X):
        eps = 1e-10 
        log_feature_probs = np.log(self.feature_probs.T.clip(eps, 1))
        log_probs = X @ log_feature_probs + self.class_priors
        # print(log_probs)
        return np.argmax(log_probs, axis=1)

    def predict_test(self, X_test, unseen_word_count):
        eps = 1e-10 
        log_feature_probs = np.log(self.feature_probs.T.clip(eps, 1))
        log_probs = X_test @ log_feature_probs + self.class_priors
        for i in range(X_test.shape[0]): 
            for c in range(len(self.class_priors)): 
                if unseen_word_count[i] > 0:
                    unseen_contribution = unseen_word_count[i] * np.log(self.unseen_word_value_per_class[c])
                    log_probs[i, c] += unseen_contribution 
        
        # print(log_probs)
        return np.argmax(log_probs, axis=1)


model = MultinomialNaiveBayes()
model.fit(X, Y, class_count)

predictions = model.predict(X)

accuracy = np.mean(predictions == Y)
# print(f'Accuracy: {accuracy * 100:.2f}%')

test_df= pd.read_csv(test_data_path,header=None,sep='\t',quoting=3)
test_x= test_df[2]
test_y= test_df[1]

def create_test_feature_matrix(processed_texts, word_to_index):
    vocab_size = len(word_to_index)
    
    feature_matrix = np.zeros((len(processed_texts), vocab_size), dtype=int)
    unseen_word_count = np.zeros(len(processed_texts), dtype=int)
    
    for i, text in enumerate(processed_texts):
        for word in text:
            if word in word_to_index:
                feature_matrix[i, word_to_index[word]] += 1  # Count occurrences of each word
            else:
                unseen_word_count[i] += 1
    return feature_matrix, unseen_word_count

test_x= test_x.apply(preprocess_text)
X_test, unseen_word_count = create_test_feature_matrix(test_x, word_to_index)

test_predictions = model.predict_test(X_test, unseen_word_count)
test_predictions = [index_to_label[pred] for pred in test_predictions]
test_accuracy = np.mean(test_predictions == test_y)
# print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

with open(output_path, 'w') as f:
    for label in test_predictions:
        f.write(f"{label}\n")
