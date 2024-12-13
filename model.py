# importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt

# import the dataset
df = pd.read_csv('creditcard.csv')

# inspect the dataset
print("Dataset shape:", df.shape)
print("Dataset info:", df.info())

# Data Preprocessing
# step 1: check for the null values
print("Null values per column:\n", df.isnull().sum())

if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print("Null values handled. Dataset shape after handling null values:", df.shape)

# step 2: split the dataset into features(X) and target(y)
X = df.drop('Class', axis=1)
y = df['Class']

# step 3: Standardization of Time & Amount columns
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
print("Time & Amount columns standardized")

# step 4: SMOTE oversampling for handling the class imbalance of 1 & 0 classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("SMOTE applied. Resampled data shape:", X_resampled.shape)

# step 5: Splitting the dataset in training & testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Train Test split completed!")

# Model Selection & Training
# part 1: Logistic Regression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# prediction
y_pred_lr = logistic_model.predict(X_test)


# part 2: Random Forest
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# prediction
y_pred_rf = random_forest_model.predict(X_test)


# part 3: Support Vector Machine
support_vector_model = SVC(random_state=42, probability=True)
support_vector_model.fit(X_train, y_train)

# prediction
y_pred_svc = support_vector_model.predict(X_test)

# Model Evaluation


def evaluate_model(model_name, y_test, y_pred, y_proba=None):
    print(f"\n--- {model_name} Evaluation ---")
    # Accuracy, Precision, Recall, F1-Score
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ROC-AUC Score
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        print("ROC-AUC Score:", roc_auc)

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend(loc="lower right")
        plt.show()


# Logistic Regression
y_proba_lr = logistic_model.predict_proba(X_test)[:, 1]
evaluate_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr)

# Random Forest
y_proba_rf = random_forest_model.predict_proba(X_test)[:, 1]
evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf)

# Support Vector Machine
y_proba_svc = support_vector_model.predict_proba(X_test)[:, 1]
evaluate_model("Support Vector Machine", y_test, y_pred_svc, y_proba_svc)
