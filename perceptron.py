from sklearn.datasets import load_svmlight_file
from urllib.request import urlopen
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # Import Matplotlib

# Load data
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes"
with urlopen(url) as f:
    X, y = load_svmlight_file(f)

# replace 0 with median
X = X.toarray()
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#split data 20:80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create Perceptron class
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
        self.accuracy_values = []  # To store accuracy values

    def activation(self, x):
        return 1 if x >= 0 else -1

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

            # Calculate and store accuracy on the validation set at the end of each epoch
            accuracy = self.calculate_accuracy(X_test, y_test)
            self.accuracy_values.append(accuracy)

    def calculate_accuracy(self, X, y):
        correct_num = 0
        data_num = X.shape[0]
        for i in range(data_num):
            x_sample = np.insert(X[i], 0, 1)
            y_pred = self.predict(x_sample)
            if y_pred == y[i]:
                correct_num += 1
        return correct_num / data_num

# number of feature
feature_num = X_train.shape[1]

# Training
perceptron = Perceptron(input_size=feature_num, lr=0.1, epochs=50000)
perceptron.fit(X_train, y_train)
print(perceptron.W)

# Correct rate on test data
accuracy = perceptron.calculate_accuracy(X_test, y_test)
print(f"Correct Rate (Accuracy) on Test Data: {accuracy * 100:.2f}%")

# Plot the learning curve
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, perceptron.epochs + 1), perceptron.accuracy_values, marker='o', linestyle='-')
# plt.title('Learning Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()
