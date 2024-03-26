
import os
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
Loading the Data
def load_data():
print("Loading data...")

ham_files_location = os.listdir("dataset/ham")
spam_files_location = os.listdir("dataset/spam")
data = []
# Load ham email
for file_path in ham_files_location:
f = open("dataset/ham/" + file_path, "r")
text = str(f.read())
data.append([text, "ham"])
# Load spam email
for file_path in spam_files_location:
f = open("dataset/spam/" + file_path, "r")
text = str(f.read())
data.append([text, "spam"])
data = np.array(data)
print("flag 1: loaded data")
return data
Data Pre-processing

# Preprocessing data: noise removaldef preprocess_data(data):
    print("Preprocessing data...")
    
    punc = string.punctuation           # Punctuation list
    sw = stopwords.words('english')     # Stopwords list
for record in data:
# Remove common punctuation and symbols
for item in punc:
record[0] = record[0].replace(item, "")
# Lowercase all letters and remove stopwords 
        splittedWords = record[0].split()
        newText = ""
        for word in splittedWords:
            if word not in sw:
                word = word.lower()
                newText = newText + " " + word          record[0] = newText
        
    print("flag 2: preprocessed data")        
    return data

#Use split() method on email text record[0] to return a list of all words in the email. Iterate over that list of words, and if the word is not in stopwords list, set it to lowercase, and add the word to newText. newText will contain the email but empty of stopwords. newText is assigned back to record[0]. After every record[0] is preprocessed, the clean data is returned.
Splitting the Data into Training and Testing Sets
The data set is split into a training set (73%) and a testing set (27%).

# Splitting original dataset into training dataset and test datasetdef split_data(data):
    print("Splitting data...")
    
    features = data[:, 0]   # array containing all email text bodies
    labels = data[:, 1]     # array containing corresponding labels
    print(labels)
    training_data, test_data, training_labels, test_labels =\
        train_test_split(features, labels, test_size = 0.27, random_state = 42)
    
    print("flag 3: splitted data")
    return training_data, test_data, training_labels, test_labels

The KNN Algorithm

def get_count(text):
     wordCounts = dict()
     for word in text.split():
          if word in wordCounts:
               wordCounts[word] += 1
         else:
               wordCounts[word] = 1

     return wordCounts
def euclidean_difference(test_WordCounts, training_WordCounts):
total = 0
for word in test_WordCounts:
    if word in test_WordCounts and word in training_WordCounts:
         total += (test_WordCounts[word] - training_WordCounts[word])**2
            del training_WordCounts[word]
else:
     total += test_WordCounts[word]**2
for word in training_WordCounts:
            total += training_WordCounts[word]**2
    return total**0.5
get_class() function
def get_class(selected_Kvalues):
    spam_count = 0
    ham_count = 0
    for value in selected_Kvalues:
        if value[0] == "spam":
            spam_count += 1
        else:
            ham_count += 1
    if spam_count > ham_count:
        return "spam"
    else:
        return "ham"
knn_classifier() function
def knn_classifier(training_data, training_labels, test_data, K, tsize):
    print("Running KNN Classifier...")
    
    result = []
    counter = 1
    # word counts for training email
    training_WordCounts = [] 
    for training_text in training_data:
            training_WordCounts.append(get_count(training_text))
    for test_text in test_data:
        similarity = [] # List of euclidean distances
        test_WordCounts = get_count(test_text)  # word counts for test email
        # Getting euclidean difference 
        for index in range(len(training_data)):
            euclidean_diff =\
                euclidean_difference(test_WordCounts, training_WordCounts[index])
            similarity.append([training_labels[index], euclidean_diff])
        # Sort list in ascending order based on euclidean difference
        similarity = sorted(similarity, key = lambda i:i[1])
        # Select K nearest neighbours
        selected_Kvalues = [] 
        for i in range(K):
            selected_Kvalues.append(similarity[i])
        # Predicting the class of email
        result.append(get_class(selected_Kvalues))
    return result
main() function
def main(K):
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    tsize = len(test_data)
    result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize) 
    accuracy = accuracy_score(test_labels[:tsize], result)
    print("training data size\t: " + str(len(training_data)))
    print("test data size\t\t: " + str(len(test_data)))
    print("K value\t\t\t\t: " + str(K))
    print("Samples tested\t\t: " + str(tsize))
    print("% accuracy\t\t\t: " + str(accuracy * 100))
    print("Number correct\t\t: " + str(int(accuracy * tsize)))
    print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))
main(11)

