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

credit_scores = train_df[[8, 9, 10, 11, 12]].apply(pd.to_numeric, errors='coerce').values
row_sums = credit_scores.sum(axis=1, keepdims=True)
credit_scores_normalized = np.where(row_sums > 0, credit_scores / row_sums, 0)

label_to_index = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true': 5}
index_to_label = {v: k for k, v in label_to_index.items()}
Y = np.array([label_to_index[label] for label in train_y])

def softmax(logits):
    exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True)) 
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def compute_loss(probs, y_true, num_classes):
    y_true_one_hot = np.eye(num_classes)[y_true] 
    log_probs = -np.log(probs[range(len(y_true)), y_true])  
    loss = np.sum(log_probs) / len(y_true)
    return loss

class LogisticRegression:
    def __init__(self, num_features, num_classes, lr=0.01):
        self.weights = np.random.randn(num_features, num_classes) * 0.01  
        self.bias = np.zeros((1, num_classes)) 
        self.lr = lr 
    
    def fit(self, X, y, num_classes, epochs):
        num_samples, num_features = X.shape
    
        for epoch in range(epochs):
            logits = np.dot(X, self.weights) + self.bias 
            probs = softmax(logits)  
            loss = compute_loss(probs, y, num_classes)

            y_true_one_hot = np.eye(num_classes)[y]
            dL_dlogits = probs - y_true_one_hot  
            dL_dw = np.dot(X.T, dL_dlogits) / num_samples 
            dL_db = np.sum(dL_dlogits, axis=0, keepdims=True) / num_samples  
            self.weights -= self.lr * dL_dw
            self.bias -= self.lr * dL_db
    
    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probs = softmax(logits)
        return np.argmax(probs, axis=1)  

num_classes = 6
num_features = 5

model = LogisticRegression(num_features=num_features, num_classes=num_classes, lr=0.01)
model.fit(credit_scores_normalized, Y, num_classes, epochs=20000)

predictions = model.predict(credit_scores_normalized)
# print("Predictions:", predictions)
accuracy = np.mean(predictions == Y)
# print(f'Accuracy: {accuracy * 100:.2f}%')

test_df= pd.read_csv(test_data_path,header=None,sep='\t',quoting=3)
test_x= test_df[2]
test_y= test_df[1]

credit_counts_test = test_df.iloc[:, 8:13].values
test_predictions = model.predict(credit_counts_test)
test_predictions = [index_to_label[pred] for pred in test_predictions]
test_accuracy = np.mean(test_predictions == test_y)
# print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

with open(output_path, 'w') as f:
    for label in test_predictions:
        f.write(f"{label}\n")