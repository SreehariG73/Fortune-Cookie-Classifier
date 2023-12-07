from collections import Counter
import numpy as np


class SimpleTextVectorizer:
    def __init__(self):
        self.vocab = {}

    def fit_transform(self, text_data):
        self.vocab = self._build_vocab(text_data)
        return self._transform(text_data)

    def transform(self, text_data):
        return self._transform(text_data)

    def _build_vocab(self, text_data):
        vocab = {}
        for doc in text_data:
            words = doc.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def _transform(self, text_data):
        num_docs = len(text_data)
        num_words = len(self.vocab)
        X = np.zeros((num_docs, num_words), dtype=int)

        for i, doc in enumerate(text_data):
            word_counts = Counter(doc.split())
            for word, count in word_counts.items():
                if word in self.vocab:
                    word_index = self.vocab[word]
                    X[i, word_index] = count

        return X


class NaiveBayesClassifierWithSmoothing:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_probs = {}
        self.word_probs = {}

    def fit(self, X, y):
        num_docs, num_words = X.shape
        self.classes = np.unique(y)

        for cls in self.classes:
            class_mask = y == cls
            num_class_docs = np.sum(class_mask)
            self.class_probs[cls] = (num_class_docs + self.alpha) / (
                num_docs + len(self.classes) * self.alpha
            )

            class_word_counts = X[class_mask].sum(axis=0)
            total_word_counts = X.sum(axis=0)
            self.word_probs[cls] = (class_word_counts + self.alpha) / (
                total_word_counts + num_words * self.alpha
            )

    def predict(self, X):
        predictions = []
        for doc in X:
            class_scores = {}
            for cls in self.classes:
                class_scores[cls] = np.log(self.class_probs[cls])
                for word_index, word_count in enumerate(doc):
                    if word_count > 0:
                        class_scores[cls] += np.log(self.word_probs[cls][word_index])
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions


stoplist = set()

with open(
    "/Users/sreehariguruprasad/myprojects/ML Assignment/Assignment 3/fortune-cookie-data/traindata.txt",
    "r",
) as file:
    train_data = file.read().splitlines()

with open(
    "/Users/sreehariguruprasad/myprojects/ML Assignment/Assignment 3/fortune-cookie-data/trainlabels.txt",
    "r",
) as file:
    train_labels = [int(label) for label in file.read().splitlines()]

with open(
    "/Users/sreehariguruprasad/myprojects/ML Assignment/Assignment 3/fortune-cookie-data/testdata.txt",
    "r",
) as file:
    test_data = file.read().splitlines()

with open(
    "/Users/sreehariguruprasad/myprojects/ML Assignment/Assignment 3/fortune-cookie-data/testlabels.txt",
    "r",
) as file:
    test_labels = [int(label) for label in file.read().splitlines()]

with open(
    "/Users/sreehariguruprasad/myprojects/ML Assignment/Assignment 3/fortune-cookie-data/stoplist.txt",
    "r",
) as file:
    for line in file:
        stoplist.add(line.strip())

clean_data = []
for i in train_data:
    words = i.split()
    filtered_words = [word for word in words if word.lower() not in stoplist]
    clean_data.append(" ".join(filtered_words))

vectorize = SimpleTextVectorizer()
X_train = vectorize.fit_transform(clean_data)
X_test = vectorize.transform(test_data)

nb = NaiveBayesClassifierWithSmoothing()
nb.fit(X_train, train_labels)
predictions = nb.predict(X_test)

from sklearn.metrics import accuracy_score

trainPredictions = nb.predict(X_train)
trainAccuracy = accuracy_score(train_labels, trainPredictions)
print(f"Training Accuracy : {trainAccuracy}")


accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy : {accuracy}")
