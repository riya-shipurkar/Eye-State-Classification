import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load CSV
df = pd.read_csv(r"C:\eye classification\EEG_Eye_State_Classification.csv")


# See the data
print(df.head())
print(df.shape)
print(df.info())       # data types and non-null counts
print(df.describe())   # stats like mean, min, max per column
# Check how many examples for each label (eye open/close)
print("Label counts (eye open/close):")
print(df.iloc[:, -1].value_counts()) #df.iloc[:, -1] means select all rows (:)
# but only the last column (-1) — which is the label column here
#.value_counts() counts how many times each unique value appears in that column.
#For your EEG eye state data, it will show how many samples have the eyes open (1) vs closed (0).

X = df.iloc[:, :-1]  # all columns except last -> features
y = df.iloc[:, -1]   # last column -> label

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 1. Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train it on the training data
model.fit(X_train, y_train)

# 3. Predict on the test data
y_pred = model.predict(X_test)

# 4. Evaluate how well it did
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model, "eye_state_model.pkl")
