import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def get_manip_data_fn():
    # Load data
    dataset = pd.read_csv('Datasets/Artificial_Neural_Networks/Titanic_Train.csv')
    x = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
    y = dataset.iloc[:, 1].values

    # Replace missing age variables with average of all ages
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imputer = imputer.fit(x[:, 2].reshape(-1, 1))
    x[:, 2:3] = imputer.transform(x[:, 2].reshape(-1, 1))

    # Replace missing boarding values with Standard
    x[[61, 829], 6] = 'S'

    # Create label encoder objects
    label_encoder_x_sex = LabelEncoder()
    label_encoder_x_embarked = LabelEncoder()

    # Encode labels to ints
    x[:, 1] = label_encoder_x_sex.fit_transform(x[:, 1])
    x[:, 6] = label_encoder_x_embarked.fit_transform(x[:, 6])

    # Encode ints as one hot
    one_hot_encoder = OneHotEncoder(categorical_features=[1])
    x = one_hot_encoder.fit_transform(x).toarray()

    # Remove first variable to avoid dummy variable trap
    x = x[:, 1:]

    # Normalize data
    standard_scalar = StandardScaler()
    x = standard_scalar.fit_transform(x)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def create_nn_function(optimizer_choice):
    # Create classifier object
    classifier = Sequential()

    # Create first layer (4 neurons, rectifier activation)
    classifier.add(Dense(output_dim=4,
                         init='uniform',
                         activation='relu',
                         input_dim=7))

    # Create second layer (4 neurons, rectifier activation)
    classifier.add(Dense(output_dim=4,
                         init='uniform',
                         activation='relu'))

    # Create final layer (1 neuron, sigmoid activation)
    classifier.add(Dense(output_dim=1,
                         init='uniform',
                         activation='sigmoid'))

    # Compile classifier (optimizer choice passed to function, cross entropy is loss function)
    classifier.compile(optimizer=optimizer_choice,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier


def calc_correct_ratio():
    # calculate correct rates
    num_of_correct = conf_matrix[0, 0] + conf_matrix[1, 1]
    num_of_incorrect = conf_matrix[0, 1] + conf_matrix[1, 0]
    total = num_of_correct + num_of_incorrect
    ratio = num_of_correct/total
    return ratio


# Fetch data
x_train, x_test, y_train, y_test = get_manip_data_fn()

# Fetch model
model = create_nn_function('adam')

# Fit data to model
model.fit(x=x_train, y=y_train, batch_size=10, nb_epoch=100)

# Predict on test data
y_prediction = model.predict(x_test)
y_prediction = (y_prediction > 0.5)

# Fit data to confusion matrix
conf_matrix = confusion_matrix(y_test, y_prediction)
# correct predictions are at 0,0 and 1,1
# incorrect predictions are at 0,1 and 1,0

# Calculate correct ratio
correct_ratio = calc_correct_ratio()

print(correct_ratio)