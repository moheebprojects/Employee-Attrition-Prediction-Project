# Employee-Attrition-Prediction-Project
## Project Overview
This project focuses on analyzing employee data and building a predictive model to help the Human Resources (HR) department at Salifort Motors, a large consulting firm, understand the factors contributing to employee attrition. The aim is to gain actionable insights into employee turnover, identify high-risk employees who might leave, and offer recommendations to improve employee retention.

## Project Goals
Perform exploratory data analysis (EDA) to uncover trends and patterns related to employee attrition.
Build and evaluate a machine learning model to predict whether an employee will leave the company.
Visualize key insights to facilitate data-driven decisions by HR.

## Dataset
The dataset contains 15,000 rows and 10 columns, each representing different employee-related variables:

satisfaction_level: Employee's self-reported satisfaction level [0–1]
last_evaluation: Score from the employee's most recent performance review [0–1]
number_project: Number of projects the employee contributes to
average_monthly_hours: Average monthly hours worked by the employee
tenure: Years spent with the company
work_accident: Whether the employee experienced a work accident (0: No, 1: Yes)
left: Whether the employee left the company (0: Stayed, 1: Left)
promotion_last_5years: Whether the employee was promoted in the last five years
department: The employee's department
salary: Employee's salary level (low, medium, high)

## Code Summary
### Data Cleaning & Preparation
Missing Values and Duplicates: Checked for and handled missing values and duplicates in the data.
Outlier Detection: Used a boxplot on the tenure column to detect and handle outliers.
Feature Renaming: Renamed some columns for clarity (e.g., "time_spend_company" to "tenure").

## Exploratory Data Analysis (EDA)
The EDA includes detailed visualizations to explore factors related to attrition, such as:

Employee Status: Proportion of employees who stayed vs. those who left, visualized with a pie chart.
Tenure: Distribution of tenure across employees who left vs. stayed.
Satisfaction Level: Histogram of satisfaction levels for employees who stayed vs. those who left.
Monthly Hours: Distribution of average monthly hours worked by both groups.
Project Count and Salary Level: Proportion of employees who left or stayed by the number of projects and salary brackets.
Retention Rate Over Time: Retention rates visualized by tenure length.
Department Analysis: Attrition rates by department and reasons related to lack of promotions or work accidents.

## Machine Learning Model
The project uses a classification model to predict whether an employee will leave based on various features. Future steps include:

Feature Selection: Based on EDA findings and multicollinearity checks.
Model Training and Evaluation: Training a logistic regression or decision tree classifier and evaluating it with metrics like accuracy, recall, and ROC-AUC.

## Results and Insights
The findings from EDA and modeling will be shared in a one-page summary with actionable insights for HR:

High Attrition Factors: Low satisfaction, high average monthly hours, and lack of promotions are strongly associated with employee turnover.
Departmental Trends: Certain departments show higher attrition, indicating areas for HR intervention.

