"""
Protein-Based Disease Classification Model

This script implements a deep learning model for binary classification of 
MASH (Metabolic dysfunction-Associated Steatohepatitis) vs Healthy Controls 
based on protein expression data.

Author: A.Leszczynska
Date: 2024
"""

# ============================================================================
# 1. SETUP AND INSTALLATION
# ============================================================================
pip install tensorflow==2.14.0 keras==2.14.0



# ============================================================================
# 2. IMPORT LIBRARIES
# ============================================================================
pip install ml-dtypes==0.2.0


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score



# ============================================================================
# 3. DATA LOADING AND EXPLORATION
# ============================================================================
data = pd.read_csv('data_file_training.csv')
data.head(31)

data.shape

featuresScale = np.array(data[['HP','HPX','SERPINA7','SERPINA3','GSTA2','PTPA','FABPA','CLC1B','FRIH','ARG1','IGFbp','TIMP2','FYN','IL27RA','OSMR','PPBP','S100A4','SAA1','ANKRD']])
print(type(featuresScale))
print(featuresScale[0:5])
print(featuresScale.shape)



# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
# Print response variable
# Labels data class values
labels = np.array(data['CLASS'])
print(type(labels))
labels[0:20]


# Encode response variable as one-hot
# Pull out labels data
# Get dummies in pandas that can do conversion one-hot
# Pandas gives in dataframe and then convert in to array nparray

label_one_hot_df = pd.get_dummies(labels)
one_hot = np.array(label_one_hot_df)
print(one_hot[0:5,:])

# Second neuron in NN indicates that person has a disease [1 0] 1=True, 0=False


# Split dataset into training + testing (33%)
# We are only taking the data that will be used for training only
print(type(featuresScale))
print(featuresScale.shape)
print(type(one_hot))
print(one_hot.shape)
train_feats = featuresScale
train_lab = one_hot




print(type(train_feats))
print(train_feats.shape)
print(type(train_lab))
print(train_lab.shape)



# ============================================================================
# 5. MODEL ARCHITECTURE
# ============================================================================
# Print Training + Testing data
# To confirm everything is okay
print("Training Predictor Vars")
print(type(train_feats))
print(train_feats.shape)
print(train_feats[:5,:1])

print("\n\nTraining Response Vars")
print(type(train_lab))
print(train_lab.shape)
print(train_lab[:5])


# Create the Neural Network Model
# Input has
# Layer size 19
# Output has two neurons [01] has a disease [10] no disease. First neuron becomes one
# Person does not have disease if second neuron is one person has a disease
feat_shape = train_feats.shape[1]
print(feat_shape)
hidden_nodes = 7
out_shape = train_lab.shape[1]
print(out_shape)


# ============================================================================
# 6. MODEL TRAINING
# ============================================================================
# Keras to build the NN load libraries and model
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Activation
print ('keras_version: ', ks.__version__)

# 2 different activation functions
# Hidden ReLU
# Output layer softmax - need to compute probability of each of classes (species)
# Build the model in KERAS
model = Sequential()
model.add(Dense(hidden_nodes, activation='relu', input_dim=feat_shape))
model.add(Dense(out_shape, activation= 'softmax'))
# Build NN
# Compile model

model.compile(loss= 'categorical_crossentropy', optimizer= 'adam')
model.summary()



# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================
# Train the model
# Feed in train features - training features and feed in train labs means one-hot variable
#
print(train_feats.shape)
print(train_lab.shape)

epochs = 20000

hist = model.fit(train_feats, train_lab, epochs=epochs, batch_size= 128, verbose= 0)


import matplotlib.pyplot as plt
train_loss = hist.history ['loss']
xc = range (epochs)
plt.plot(xc, train_loss)

plt.figure(1,figsize=(7,5))
plt.xlabel('num of epochs')
plt.ylabel('loss')
plt.title('train_loss')
plt.grid(True)
plt.style.use(['ggplot'])
# Loss function soon becomes zero means model has converged

# Print values of all the weights and bias of all the neurons of NN
#

for layerNum, layer in enumerate(model.layers):
  print('Layer Number = ', layerNum)
####
  print("Weights Values =")
  weights = layer.get_weights()[0]
  print(weights)
####
  print("Bias Values = ")
  biases = layer.get_weights()[1]
  print(biases)
  print("==========================")




# Predictions
# Take the first observation from the train dataset to test the model.

# model.predict(train_feats[0:1], batch_size=None, verbose=0 steps=None)
test_feats2 = np.array([[-0.3723,-0.44599, -0.41352, -0.42817, -0.46598, -0.43858, -0.4114, -0.37505, -0.33182, -0.08496, -0.44128,-0.21554, 2.213466, 2.213466, -0.42611, -0.36788, -0.39346, -0.37888, -0.47216]])
print(type(test_feats2))
print(test_feats2.shape)
print(test_feats2[:5,:])

#######
print("Test2 Response Vars")
test_lab2 = np.array([[1,0]])
print(type(test_lab2))
print(test_lab2.shape)
print(test_lab2[:5])




# Assuming you have preprocessed your test data in the same way as your training data
# Make predictions on the test data (test_feats2)
# This one refers to the values I input manually
predictions = model.predict(test_feats2)

# If you have multiple test samples, you can use the following:
# predictions = model.predict(test_feats)

print("Predictions:")
print(predictions)

# If you want to get the predicted class labels (assuming you have one-hot encoded your output):
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted Classes:")
print(predicted_classes)


import pandas as pd
import numpy as np

# Load the CSV file with test data into a DataFrame
test_data = pd.read_csv('data_file_Testing_1.csv')

# Assuming you have preprocessed your test data as needed (scaling, encoding, etc.)
# Ensure that the preprocessing steps match what you did with your training data

# Check the number of features in your test data
num_features_test = test_data.shape[1]

# Print the summary of your model to check the input size of the first layer
model.summary()

# Assuming the first layer is a Dense layer, check its input size
input_size_model = model.layers[0].input_shape[1]

if num_features_test != input_size_model:
    print(f"Error: Number of features in test data ({num_features_test}) does not match the expected input size ({input_size_model}) of the model.")
else:
    # Make predictions on the preprocessed test data
    predictions = model.predict(test_data)

    # If you have multiple test samples, you can use the following:
    # predictions = model.predict(preprocessed_test_data)

    print("Predictions (Probabilities):")
    print(predictions)

    # If you want to get the predicted class labels (assuming you have one-hot encoded your output):
    predicted_classes = np.argmax(predictions, axis=1)
    print("Predicted Classes:")
    print(predicted_classes)


import pandas as pd

# Load the CSV file with test data into a DataFrame
test_data = pd.read_csv('data_file_Testing_1.csv')

# Make predictions on the preprocessed test data
predictions = model.predict(test_data)

# Get the predicted class labels
predicted_classes = np.argmax(predictions, axis=1)

# Print the predicted classes
print("Predicted Classes:")
print(predicted_classes)


import pandas as pd

# Load the CSV file with test data into a DataFrame
test_data = pd.read_csv('./BA_NORMtoTakeda_Testing_1.csv')

# Make predictions on the preprocessed test data
predictions = model.predict(test_data)

# Get the predicted class labels
predicted_classes = np.argmax(predictions, axis=1)

# Add the predicted classes as a new column to the DataFrame
test_data['Predicted Classes'] = predicted_classes

# Print the DataFrame with the predicted classes
print("Test Data with Predicted Classes:")
print(test_data)


from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Provided true labels and predicted classes
y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred_binary = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

# Compute and print classification report
report = classification_report(y_true, y_pred_binary)
print("Classification Report:")
print(report)

# Save classification report to a text file
with open('classification_report.txt', 'w') as file:
    file.write("Classification Report:\n")
    file.write(report)

# Save classification report as a PDF
plt.figure(figsize=(8, 6))
plt.text(0.1, 0.5, report, fontsize=10, ha='left', va='center', linespacing=1.5)
plt.axis('off')
plt.tight_layout()
plt.savefig('classification_report.pdf')
plt.show()




pip install scikit-learn==1.2.2  # Replace with the desired version if needed


!pip install --upgrade scikit-learn

pip install --upgrade matplotlib scikit-learn


pip install numpy==1.24.3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import numpy as np
from sklearn.metrics import confusion_matrix

# Given data
y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred_probs = np.array([[1.2861042e-22, 9.9999994e-01],
                         [1.7898401e-38, 9.9999994e-01],
                         [3.0513549e-14, 9.9999994e-01],
                         [2.1470170e-16, 9.9999994e-01],
                         [4.1513424e-14, 9.9999994e-01],
                         [1.9189721e-13, 9.9999994e-01],
                         [2.8100292e-16, 9.9999994e-01],
                         [1.2005161e-13, 9.9999994e-01],
                         [7.5021582e-21, 9.9999994e-01],
                         [4.8174597e-22, 9.9999994e-01],
                         [2.1022509e-13, 9.9999994e-01],
                         [1.3404570e-15, 9.9999994e-01],
                         [2.1289189e-13, 9.9999994e-01],
                         [4.0697961e-19, 9.9999994e-01],
                         [4.8709666e-14, 9.9999994e-01],
                         [2.9079214e-14, 9.9999994e-01],
                         [5.3633215e-13, 9.9999994e-01],
                         [1.4725979e-14, 9.9999994e-01],
                         [9.9999994e-01, 3.6428807e-10],
                         [9.9999994e-01, 2.0020020e-10],
                         [9.9999994e-01, 4.8375678e-15],
                         [9.9999994e-01, 3.1811670e-10],
                         [9.9999994e-01, 2.3875904e-08],
                         [9.9999994e-01, 1.1156861e-09],
                         [9.9999994e-01, 1.9577605e-10],
                         [9.9999994e-01, 1.3941356e-09],
                         [9.9999994e-01, 2.0230035e-09],
                         [9.9999994e-01, 2.3887448e-10],
                         [2.4580644e-04, 9.9975431e-01],
                         [9.9999994e-01, 1.4636836e-12],
                         [9.9999970e-01, 2.3731715e-07],
                         [9.9999994e-01, 5.5778823e-11],
                         [9.9999118e-01, 8.8651223e-06],
                         [1.0000000e+00, 5.2630938e-11],
                         [1.0000000e+00, 1.4389376e-11],
                         [1.0000000e+00, 3.4384615e-10],
                         [9.9999988e-01, 1.6236544e-07]])
# Set your threshold
threshold = 0.97

# Convert predicted probabilities to binary predictions based on the threshold
y_pred = (y_pred_probs[:, 1] >= threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Extract TP, TN, FP, FN
TN, FP, FN, TP = cm.ravel()

# Display the results
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")


# Good Confusion Matrix with the data fed in np array


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Given data
y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred_probs = np.array([[1.2861042e-22, 9.9999994e-01],
                         [1.7898401e-38, 9.9999994e-01],
                         [3.0513549e-14, 9.9999994e-01],
                         [2.1470170e-16, 9.9999994e-01],
                         [4.1513424e-14, 9.9999994e-01],
                         [1.9189721e-13, 9.9999994e-01],
                         [2.8100292e-16, 9.9999994e-01],
                         [1.2005161e-13, 9.9999994e-01],
                         [7.5021582e-21, 9.9999994e-01],
                         [4.8174597e-22, 9.9999994e-01],
                         [2.1022509e-13, 9.9999994e-01],
                         [1.3404570e-15, 9.9999994e-01],
                         [2.1289189e-13, 9.9999994e-01],
                         [4.0697961e-19, 9.9999994e-01],
                         [4.8709666e-14, 9.9999994e-01],
                         [2.9079214e-14, 9.9999994e-01],
                         [5.3633215e-13, 9.9999994e-01],
                         [1.4725979e-14, 9.9999994e-01],
                         [9.9999994e-01, 3.6428807e-10],
                         [9.9999994e-01, 2.0020020e-10],
                         [9.9999994e-01, 4.8375678e-15],
                         [9.9999994e-01, 3.1811670e-10],
                         [9.9999994e-01, 2.3875904e-08],
                         [9.9999994e-01, 1.1156861e-09],
                         [9.9999994e-01, 1.9577605e-10],
                         [9.9999994e-01, 1.3941356e-09],
                         [9.9999994e-01, 2.0230035e-09],
                         [9.9999994e-01, 2.3887448e-10],
                         [2.4580644e-04, 9.9975431e-01],
                         [9.9999994e-01, 1.4636836e-12],
                         [9.9999970e-01, 2.3731715e-07],
                         [9.9999994e-01, 5.5778823e-11],
                         [9.9999118e-01, 8.8651223e-06],
                         [1.0000000e+00, 5.2630938e-11],
                         [1.0000000e+00, 1.4389376e-11],
                         [1.0000000e+00, 3.4384615e-10],
                         [9.9999988e-01, 1.6236544e-07]])

# Set your threshold
threshold = 0.97

# Convert predicted probabilities to binary predictions based on the threshold
y_pred = (y_pred_probs[:, 1] >= threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['MASH','HC'])  # Replace [0, 1] with your class labels
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
