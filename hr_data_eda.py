import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(".../HR_comma_sep2.csv")

# Ensure all columns are displayed
pd.set_option("display.max_columns", None)

df.info()

# Gather basic information about the data
df.head(n=10)
# print(df.head(n=10))
# print(df.head(5))

# Gather descriptive statistics about the data
df.describe()
# print(df.describe())


# Display all column names
df.columns
# print(df.columns)

# Rename columns as needed
df = df.rename(
    columns={
        "time_spend_company": "tenure",
        "Work_accident": "work_accident",
        "Department": "department",
        "average_montly_hours": "average_monthly_hours",
    }
)

# Display all column names after the update
df.columns
# print(df.columns)

# Check for missing values
df.isna().sum()

# Check for duplicates
df.duplicated().sum()
# Inspect some rows containing duplicates as needed
duplicates = df[df.duplicated()].head(n=10)
print(duplicates)

# Drop duplicates and save resulting dataframe in a new variable as needed
df_clean = df.drop_duplicates()

# Display first few rows of new dataframe as needed
df_clean.head(n=10)

# Saving clean Dataframe
df_clean.to_csv(
    ".../hr_cleaned_data.csv",
    index=False,
)

# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x=df_clean["tenure"])
plt.xlabel("Tenure")
plt.title("Distribution of Tenure")
plt.show()

# Determine the number of rows containing outliers
# Define threshold for outliers (e.g., outside 1.5 times the interquartile range)
threshold = 1.5

# Calculate the interquartile range (IQR)
q1 = df_clean["tenure"].quantile(0.25)
q3 = df_clean["tenure"].quantile(0.75)
iqr = q3 - q1

# Calculate the lower and upper bounds for outliers
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr

# Count the number of rows containing outliers
outliers_count = df_clean[
    (df_clean["tenure"] < lower_bound) | (df_clean["tenure"] > upper_bound)
].shape[0]

# Display the number of rows containing outliers
print("Number of rows containing outliers in 'tenure':", outliers_count)

# Get numbers of people who left vs. stayed
left_counts = df_clean["left"].value_counts()

# Get percentages of people who left vs. stayed
left_percentages = df_clean["left"].value_counts(normalize=True) * 100

# Display the results
print("Number of employees who left vs. stayed:")
print(left_counts)
print("\nPercentage of employees who left vs. stayed:")
print(left_percentages)


# Calculate the count of employees who left versus stayed
attrition_counts = df_clean["left"].value_counts()

# Define labels and colors for the pie chart
labels = ["Stayed", "Left"]
colors = ["#2ecc71", "#e74c3c"]

# Create the advanced pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    attrition_counts,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    explode=(0, 0.1),
)
plt.title("Distribution of Employees: Stayed vs. Left")

# Add legend
plt.legend(title="Employee-", labels=["Stayed (0)", "Left (1)"])

plt.axis("equal")
plt.show()


# Plot histogram for tenure
plt.figure(figsize=(8, 6))
sns.histplot(
    df_clean[df_clean["left"] == 0]["tenure"], kde=True, color="green", label="Stayed"
)
sns.histplot(
    df_clean[df_clean["left"] == 1]["tenure"], kde=True, color="red", label="Left"
)
plt.xlabel("Tenure")
plt.ylabel("Count")
plt.title("Histogram of Tenure")
plt.legend()
plt.show()


# Plot histogram for satisfaction_level
plt.figure(figsize=(8, 6))
sns.histplot(
    df_clean[df_clean["left"] == 0]["satisfaction_level"],
    kde=True,
    color="green",
    label="Stayed",
)
sns.histplot(
    df_clean[df_clean["left"] == 1]["satisfaction_level"],
    kde=True,
    color="red",
    label="Left",
)
plt.xlabel("Satisfaction Level")
plt.ylabel("Count")
plt.title("Histogram of Satisfaction Level")
plt.legend()
plt.show()

# Plot histogram for last_evaluation
plt.figure(figsize=(8, 6))
sns.histplot(
    df_clean[df_clean["left"] == 0]["last_evaluation"],
    kde=True,
    color="green",
    label="Stayed",
)
sns.histplot(
    df_clean[df_clean["left"] == 1]["last_evaluation"],
    kde=True,
    color="red",
    label="Left",
)
plt.xlabel("Last Evaluation")
plt.ylabel("Count")
plt.title("Histogram of Last Evaluation")
plt.legend()
plt.show()

# Plot histogram for average_monthly_hours
plt.figure(figsize=(8, 6))
sns.histplot(
    df_clean[df_clean["left"] == 0]["average_monthly_hours"],
    kde=True,
    color="green",
    label="Stayed",
)
sns.histplot(
    df_clean[df_clean["left"] == 1]["average_monthly_hours"],
    kde=True,
    color="red",
    label="Left",
)
plt.xlabel("Average Monthly Hours")
plt.ylabel("Count")
plt.title("Histogram of Average Monthly Hours")
plt.legend()
plt.show()

# Plot histogram for number_project
plt.figure(figsize=(8, 6))
sns.histplot(
    df_clean[df_clean["left"] == 0]["number_project"],
    kde=True,
    color="green",
    label="Stayed",
)
sns.histplot(
    df_clean[df_clean["left"] == 1]["number_project"],
    kde=True,
    color="red",
    label="Left",
)
plt.xlabel("Number of Projects")
plt.ylabel("Count")
plt.title("Histogram of Number of Projects")
plt.legend()
plt.show()


# Create bins for different ranges of last evaluation scores
bins = pd.cut(
    df_clean["last_evaluation"],
    bins=[0, 0.5, 0.75, 1],
    labels=["Low", "Medium", "High"],
)

# Calculate the attrition rate for each bin
attrition_rate = df_clean.groupby(bins)["left"].mean() * 100

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette="RdYlGn")
plt.xlabel("Last Evaluation Score")
plt.ylabel("Attrition Rate (%)")
plt.title("Attrition Rate by Last Evaluation Score")

# Add count labels to the bars
for i, value in enumerate(attrition_rate.values):
    plt.text(i, value, f"{value:.2f}%", ha="center")

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

plt.show()

# Calculate the proportion of employees who left in each department
left_by_department = df_clean[df_clean["left"] == 1]["department"].value_counts(
    normalize=True
)

# Sort the departments based on the proportion of employees who left
left_by_department = left_by_department.sort_values(ascending=False)

# Create the advanced horizontal bar plot
plt.figure(figsize=(10, 6))
colors = sns.color_palette("viridis", len(left_by_department))
ax = sns.barplot(
    x=left_by_department.values, y=left_by_department.index, palette=colors
)
plt.xlabel("Proportion of Employees who Left")
plt.ylabel("Department")
plt.title("Proportion of Employees who Left by Department")

# Add data labels to the bars
for i, value in enumerate(left_by_department.values):
    ax.text(value, i, f"{value:.2%}", va="center")

plt.show()

plt.figure(figsize=(10, 6))

# Create scatter plot comparing employees who stayed versus those who left
sns.scatterplot(
    x="average_monthly_hours",
    y="satisfaction_level",
    hue="left",
    data=df_clean,
    palette={0: "#2ecc71", 1: "#e74c3c"},
    alpha=0.6,
)

# Set labels and title
plt.xlabel("Average Monthly Hours")
plt.ylabel("Satisfaction Level")
plt.title(
    " Satisfaction Level vs Average Monthly Hours\nfor Employees who Stayed vs. Left"
)

# Add legend
plt.legend(title="Employee-", labels=["Stayed (0)", "Left (1)"])
plt.show()


plt.figure(figsize=(10, 6))

# Calculate the proportion of employees who left versus stayed based on project counts
project_counts = df_clean.groupby(["number_project", "left"]).size().unstack()
project_counts["Total"] = project_counts.sum(axis=1)
project_proportions = project_counts.div(project_counts["Total"], axis=0)

# Create the stacked bar plot
project_proportions[[0, 1]].plot(
    kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], alpha=0.8
)

# Set labels and title
plt.xlabel("Number of Projects")
plt.ylabel("Proportion")
plt.title("Proportion of Employees who Left vs. Stayed\nBased on Number of Projects")

# Add legend
plt.legend(title="Employee-", labels=["Stayed (0)", "Left (1)"])
plt.show()
plt.figure(figsize=(10, 6))

# Count the number of employees in each salary bracket
salary_count = df_clean.groupby(["salary", "left"]).size().unstack()

# Set the colors for employees who stayed and left
colors = ["#2ecc71", "#e74c3c"]

# Create the grouped bar plot
salary_count.plot(kind="bar", stacked=True, color=colors)

# Add labels, title, and legend
plt.xlabel("Salary Bracket")
plt.ylabel("Count")
plt.title("Count of Employees in Different Salary Brackets by Retention Status")
plt.legend(title="Employee-", labels=["Stayed (0)", "Left (1)"])

# Add percentage labels on top of each bar
total_counts = salary_count.sum(axis=1)
for i, (index, row) in enumerate(salary_count.iterrows()):
    for j, count in enumerate(row):
        percentage = count / total_counts[i] * 100
        plt.text(
            i, row[:j].sum() + count / 2, f"{percentage:.1f}%", ha="center", va="center"
        )

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

# Group the data by tenure and calculate the turnover rate
turnover_rate = df_clean.groupby("tenure")["left"].mean()

# Create the line plot
turnover_rate.plot(marker="o", linestyle="-", color="#2ecc71")
plt.xlabel("Tenure")
plt.ylabel("Employee Retention Rate")
plt.title("Employee Retention Rate over Time")
plt.legend()

# Annotate each data point with its turnover rate
for x, y in zip(turnover_rate.index, turnover_rate.values):
    plt.annotate(
        f"{y:.2%}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="black",
    )

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

# Group the data by department and left, and calculate the proportion of work accidents
work_accident_prop = (
    df_clean.groupby(["department", "left"])["work_accident"].mean().unstack()
)

# Set the order of departments
department_order = [
    "sales",
    "technical",
    "support",
    "IT",
    "product_mng",
    "marketing",
    "RandD",
    "accounting",
    "hr",
    "management",
]

# Create the grouped bar plot
work_accident_prop.plot(kind="bar", stacked=True, color=["#e74c3c", "#2ecc71"])
plt.xlabel("Department")
plt.ylabel("Proportion of Work Accidents")
plt.title("Proportion of Employees with Work Accidents by Department and Left")
plt.xticks(rotation=45)
plt.legend(["Left", "Stayed"])

# Add the proportion labels for each bar
for i, department in enumerate(department_order):
    left_prop = work_accident_prop.loc[department, 1]
    stayed_prop = work_accident_prop.loc[department, 0]
    plt.text(i, stayed_prop / 2, f"{stayed_prop:.2%}", ha="center", color="black")
    plt.text(
        i, stayed_prop + left_prop / 2, f"{left_prop:.2%}", ha="center", color="black"
    )


plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

# Group the data by department and left, and calculate the proportion of employees who left due to lack of promotions
promotion_prop = (
    df_clean[df_clean["left"] == 1]
    .groupby("department")["promotion_last_5years"]
    .mean()
    .reset_index()
)

# Set the order of departments
department_order = [
    "sales",
    "technical",
    "support",
    "IT",
    "product_mng",
    "marketing",
    "RandD",
    "accounting",
    "hr",
    "management",
]

# Create the stacked bar plot using Seaborn
sns.barplot(
    data=promotion_prop, x="department", y="promotion_last_5years", color="#e74c3c"
)

plt.xlabel("Department")
plt.ylabel("Proportion of Employees Left due to Lack of Promotions")
plt.title("Proportion of Employees Left due to Lack of Promotions by Department")
plt.xticks(rotation=45)

# Add the proportion labels for each bar
for i, row in promotion_prop.iterrows():
    prop = row["promotion_last_5years"]
    plt.text(i, prop / 2, f"{prop:.2%}", ha="center", color="black")

plt.tight_layout()
plt.show()
