import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def load_and_prepare_data(filepath):
    # Load the dataset
    df = pd.read_csv(".../hr_cleaned_data.csv")

    # Encode the 'salary' column
    df["salary"] = (
        df["salary"]
        .astype("category")
        .cat.set_categories(["low", "medium", "high"])
        .cat.codes
    )

    # Dummy encode the 'department' column
    df = pd.get_dummies(df, columns=["department"], drop_first=True)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def remove_outliers(df, column):
    """Remove outliers from a specified column using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]


def plot_heatmap(df, columns):
    """Plot a heatmap of the specified columns to visualize correlation."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[columns].corr(), vmin=-1, vmax=1, annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()


def evaluate_model(y_test, y_pred, clf):
    """Print evaluation metrics and plot the confusion matrix."""
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap="Greens")
    plt.title("Confusion Matrix")
    plt.show()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))


# Load and clean data
df_clean = load_and_prepare_data("path/to/HR_comma_sep2.csv")

# Remove outliers in 'tenure'
df_clean = remove_outliers(df_clean, "tenure")

# Visualize correlations
plot_heatmap(
    df_clean,
    [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_monthly_hours",
        "tenure",
    ],
)

# Split features and target
X = df_clean.drop(columns="left")
y = df_clean["left"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Logistic Regression Model
log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Results:")
evaluate_model(y_test, y_pred_log_reg, log_reg)

# Decision Tree with Grid Search
dt = DecisionTreeClassifier(random_state=42)
cv_params = {
    "max_depth": [2, 4, 6, 8, None],
    "min_samples_leaf": [2, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}
scoring = {"accuracy", "precision", "recall", "f1", "roc_auc"}

clf_tree = GridSearchCV(dt, cv_params, scoring=scoring, cv=5, refit="roc_auc")
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)

print("\nDecision Tree with GridSearchCV Results:")
evaluate_model(y_test, y_pred_tree, clf_tree.best_estimator_)
