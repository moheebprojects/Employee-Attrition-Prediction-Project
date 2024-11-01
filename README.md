# Employee-Attrition-Prediction-Project
## Project Overview
This project focuses on analyzing employee data and building a predictive model to help the Human Resources (HR) department at Salifort Motors, a large consulting firm, understand the factors contributing to employee attrition. The aim is to gain actionable insights into employee turnover, identify high-risk employees who might leave, and offer recommendations to improve employee retention.

## Project Goals
- Perform exploratory data analysis (EDA) to uncover trends and patterns related to employee attrition.
- Build and evaluate a machine learning model to predict whether an employee will leave the company.
- Visualize key insights to facilitate data-driven decisions by HR.

## Dataset
The dataset contains 15,000 rows and 10 columns, each representing different employee-related variables:

![image](https://github.com/user-attachments/assets/a29cf125-8054-4cab-bcce-cbf901ded578)


## Code Summary
### Data Cleaning & Preparation
- Missing Values and Duplicates: Checked for and handled missing values and duplicates in the data.
- Outlier Detection: Used a boxplot on the tenure column to detect and handle outliers.
- Feature Renaming: Renamed some columns for clarity (e.g., "time_spend_company" to "tenure").

## Exploratory Data Analysis (EDA)
The EDA includes detailed visualizations to explore factors related to attrition, such as:

- Employee Status: Proportion of employees who stayed vs. those who left, visualized with a pie chart.
- Tenure: Distribution of tenure across employees who left vs. stayed.
- Satisfaction Level: Histogram of satisfaction levels for employees who stayed vs. those who left.
- Monthly Hours: Distribution of average monthly hours worked by both groups.
- Project Count and Salary Level: Proportion of employees who left or stayed by the number of projects and salary brackets.
- Retention Rate Over Time: Retention rates visualized by tenure length.
- Department Analysis: Attrition rates by department and reasons related to lack of promotions or work accidents.

## Machine Learning Analysis Overview
#### Data Preprocessing:

Loads and cleans HR dataset.
Encodes salary as an ordinal variable and department with dummy encoding.
Checks for and drops any duplicate rows.
Removes outliers from the tenure column using the IQR method.

#### Multicollinearity Check:

Uses Variance Inflation Factor (VIF) to detect multicollinearity in the numerical features.

#### Data Visualization:

Generates a correlation heatmap of selected features to visualize potential correlations.

#### Train-Test Split:

Splits the cleaned dataset into training (75%) and testing (25%) subsets.

#### Model 1: Logistic Regression:

Trains a logistic regression classifier on the training set.
Evaluates model accuracy, precision, recall, F1 score, and confusion matrix.

#### Model 2: Decision Tree with Hyperparameter Tuning:

Initializes a Decision Tree model and tunes hyperparameters (max_depth, min_samples_leaf, and min_samples_split) using GridSearchCV.
Selects the model based on maximizing ROC-AUC and evaluates it using the same metrics as logistic regression.

#### Model Evaluation:

Outputs key evaluation metrics: accuracy, precision, recall, F1 score, and classification report.
Plots confusion matrices for each model for easy comparison.
Generates a ROC curve and calculates the ROC-AUC score for Logistic Regression.

## Results and Insights
The findings from EDA and modeling will be shared in a one-page summary with actionable insights for HR:

- High Attrition Factors: Low satisfaction, high average monthly hours, and lack of promotions are strongly associated with employee turnover.
- Departmental Trends: Certain departments show higher attrition, indicating areas for HR intervention.

