# Loan-Eligibility-Prediction-Model
Developed an 79% accurate Hist Gradient Boosting Classifier model using python programming for predicting loan eligibility of an individual. Model results were validated by employing Confusion Matrix and Graphical Verification techniques.
Code :
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from google.colab import files
from sklearn import preprocessing
import seaborn as sns
uploaded = files.upload()
dataset = pd.read_csv('LoanApprovalPrediction(3).csv', skiprows=1)

# Split Dataset into X and Y
X = dataset.iloc[:, [4, 5, 8, 9, 10, 11]].values
y = dataset.iloc[:, 12].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Perform Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Handle Missing Values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train the Classifier
classifier = HistGradientBoostingClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predict the Test Set Results
y_pred = classifier.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot the Confusion Matrix with Scale or Label Box
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y))
plt.yticks(tick_marks, np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()

# Add Color Scale or Label Box
cbar = plt.colorbar()
cbar.set_label('Number of Samples', rotation=270, labelpad=20)

# Add Text Annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.show()

# Ploting graph
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.xlabel('Test Value')
plt.ylabel('Predicted Value')
plt.show()

# Function to apply label encoding
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = preprocessing.LabelEncoder()  # Use the imported LabelEncoder class
            data[col] = le.fit_transform(data[col])

    return data

# Applying function in whole column
dataset = encode_labels(dataset)

# Generating Heatmap
sns.heatmap(dataset.corr() > 0.8, annot=True, cbar=False)
plt.show()
