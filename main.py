
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_excel("data/Dry_Bean_Dataset.xlsx")
class_counts = df["Class"].value_counts()

xaxis = class_counts.index
yaxis = class_counts.values

plt.bar(xaxis, yaxis)
plt.xlabel("Bean Variety")
plt.xticks(rotation=45)
plt.ylabel("Number of Samples")
plt.yticks(rotation=45)
plt.title("Dry Bean Class Distribution")
plt.show()

df = df.drop_duplicates()
print(df.shape)
print(df.duplicated().sum())

X = df.drop(columns=["Class"])
y = df["Class"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

param_grid = {
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_leaf": [1, 3, 5, 10]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(best_model.classes_)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

display.plot(cmap="Blues")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred))
print(X.columns)
print(best_model.feature_importances_)

importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)

sorted_importance = importance.sort_values(ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(sorted_importance.index, sorted_importance.values)
plt.xlabel("Feature Importance Score")
plt.ylabel("Bean Features")
plt.title("Decision Tree Feature Importance for Dry Bean Classification")
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 9))

plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=best_model.classes_,
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=8
)

plt.title("Dry Bean Decision Tree Visualization")
plt.show()
