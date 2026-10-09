import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

df=pd.read_csv(r"C:\Users\bharath\OneDrive\Desktop\python\healthcare_raw_dataset.csv")
new_df=df.copy()
print(new_df)

print("shape:",new_df.shape)
print(new_df.info())
print(new_df.isnull().sum())
print(new_df.describe())
print(new_df.head(10))

# inspecting duplicates

duplicate_df=df.duplicated().sum()
print("total duplicate:",duplicate_df)

duplicate_rows=new_df.duplicated(keep=False)
duplicate_rows=duplicate_rows.sort_values()
print("duplicate_row",duplicate_rows)

# null handing

print(new_df.isnull().sum())

missing_percentage=(new_df.isnull().sum()/len(new_df)*100)
print(missing_percentage)

# Fill missing categorical value
new_df["room_number"] = new_df["room_number"].fillna("Unknown")

# Convert date columns
new_df["admission_date"] = pd.to_datetime(new_df["admission_date"])
new_df["discharge_date"] = pd.to_datetime(new_df["discharge_date"])

# Check missing values after cleaning
print(new_df.isnull().sum())


# checking data type

print(new_df.dtypes)

# invalid check 

print(new_df["age"].describe())

print((new_df["age"]==0).sum())

print((new_df["age"]<0).sum())

print((new_df["age"]>100).sum())

print(new_df["gender"].value_counts())

print(new_df[new_df["gender"]=="unknown"])
print((new_df=="unknown").sum())
print(new_df.isnull().sum())

print(new_df[new_df["admission_date"]=="Unknown"])
print(new_df[new_df["admission_date"].isna()])
print(new_df["admission_date"].isna().sum())

print((new_df["discharge_date"]<new_df["admission_date"]).sum())

print(new_df[["admission_date","discharge_date"]].head(20))

# EDA

print(new_df.shape)
print(new_df.describe())

print(new_df["gender"].value_counts())
print(new_df["blood_type"].value_counts())
print(new_df["medical_condition"].value_counts())
print(new_df["admission_type"].value_counts())
print(new_df["test_result"].value_counts())
print(new_df[["age","billing_amount"]].corr())

sns.histplot(
    new_df["billing_amount"],
    bins=30,
    kde=True
)
plt.show()

avg_billing=new_df.groupby("medical_condition")["billing_amount"].mean()
print(avg_billing.sort_values(ascending=False))

sns.barplot(
    x=avg_billing.values,
    y=avg_billing.index
)
plt.xlabel("avg_billing")
plt.ylabel("medical_condition")
plt.title("avg_billing amount by medical_condition")
plt.show()

avg_billing_admission=new_df.groupby("admission_type")["billing_amount"].mean()
print(avg_billing_admission.sort_values(ascending=False))


fig=px.bar(
    x=avg_billing_admission.values,
    y=avg_billing_admission.index,
    orientation="h",
    labels={
        "x":"average billing amount",
        "y":"admission type"
    },
    title="avgerage bill by admission type"
)
fig.show()

avg_billing_test=new_df.groupby("test_result")["billing_amount"].mean()
print(avg_billing_test.sort_values(ascending=False))

fig = px.box(
    new_df,
    x="test_result",
    y="billing_amount",
    labels={
        "test_result": "Test Result",
        "billing_amount": "Billing Amount"
    },
    title="Billing Amount by Test Result"
)

fig.show()

test_billing=new_df.groupby("test_result")["billing_amount"].agg(
    ["mean","count","median","min","max"]
)
print(test_billing.sort_values("mean",ascending=False))

new_df["length_of_stay"]=(new_df["discharge_date"]-new_df["admission_date"]).dt.days

print(new_df["length_of_stay"].describe())
print((new_df["length_of_stay"]<0).sum())
print(new_df[["length_of_stay","billing_amount"]].corr())


avg_stay_admission=new_df.groupby("admission_type")["length_of_stay"].agg(
    ["count","mean","median","min","max"]
)
print(avg_stay_admission.sort_values("mean",ascending=False))


condition_stay=new_df.groupby("medical_condition")["length_of_stay"].agg(
    ["count","mean","median","min","max"]
)
print(condition_stay.sort_values("mean",ascending=False))


fig=px.scatter(
    new_df,
    x="length_of_stay",
    y="billing_amount",
    labels={
        "length_of_stay":"length of stay (days)",
        "billing_amount":"billing amount"
    },
    title="length of stay vs billing amount "
)
fig.show()

new_df["admission_month"]=new_df["admission_date"].dt.to_period("M")
monthy_admission=new_df.groupby("admission_month").size()
print(monthy_admission)

fig=px.line(
    monthy_admission,
    x=monthy_admission.index.astype(str),
    y=monthy_admission.values,
    labels={
        "x":"admission monthy",
        "y":"number of admission"
    },
    title="monthy patiend admission"
    
)
fig.show()
            
new_df["admission_year"]=new_df["admission_date"].dt.year

yearly_admission=new_df.groupby("admission_year").size()
print(yearly_admission)           

fig=px.bar(
    x=yearly_admission.index,
    y=yearly_admission.values,
    labels={
        "x": "Admission Year",
        "y": "Number of Admissions"
    },
    title="yaerly patiend admission "
) 
fig.show()  

admission_year_type=pd.crosstab(
    new_df["admission_year"],
    new_df["admission_type"]
) 

print(admission_year_type)  

fig = px.bar(
    admission_year_type,
    barmode="stack",
    labels={
        "admission_year": "Admission Year",
        "value": "Number of Admissions",
        "admission_type": "Admission Type"
    },
    title="Admission Type by Year"
)

fig.show()   

admission_year_percentage=pd.crosstab(
    new_df["admission_year"],
    new_df["admission_type"],
    normalize="index"
)*100               
print(admission_year_percentage.round(2))

fig=px.bar(
    admission_year_percentage,
    barmode="relative",
    labels={
        "admission_year": "Admission Year",
        "value": "Percentage of Admissions (%)",
        "admission_type": "Admission Type"
    },
    title="admission type percentage by year"
)
fig.show()

condition_admission=pd.crosstab(
    new_df["medical_condition"],
    new_df["admission_type"]
)
print(condition_admission)

condition_admission_percentage=pd.crosstab(
    new_df["medical_condition"],
    new_df["admission_type"],
    normalize="index"
)*100
print(condition_admission_percentage.round(2))

fig=px.imshow(
    condition_admission_percentage,
    text_auto=".2f",
    labels={
        "x": "Admission Type",
        "y": "Medical Condition",
        "color": "Percentage (%)"
    },
    title="Admission Type Percentage by Medical Condition"
)
fig.show()

medication_condition = pd.crosstab(
    new_df["medical_condition"],
    new_df["medication"]
)

# Business Insights
# 1. Patient demand remained relatively stable

# Patient admissions stayed within a relatively narrow range from 2019–2024, with 1,441 admissions in 2019 and 1,366 in 2023.

# Business insight:
# The hospital experienced a relatively stable patient volume over the analyzed period, which can support more predictable planning of staffing, beds, and other operational resources.

# 2. Certain medical conditions are associated with higher billing

# Diabetes had the highest average billing among the conditions analyzed, at approximately 5,199, followed by Hypertension (~5,066) and COPD (~5,020).

# Business insight:
# Higher average billing for certain conditions may indicate differences in treatment requirements, resource utilization, or patient-care complexity. These conditions could be examined further for cost and resource planning.

# 3. Admission type has limited impact on average billing

# Average billing across Elective, Emergency, Urgent, and Unknown admission types was relatively close.

# Business insight:
# Admission type alone does not appear to explain major differences in billing in this dataset. Other factors, such as medical condition, length of stay, medication, or treatment requirements, may be more useful for understanding billing variation.

# 4. Hospital stay duration is an important operational metric

# The project compares length of stay with billing amount using a scatter plot.

# Business insight:
# Monitoring length of stay alongside billing can help healthcare management identify patients or conditions that require longer hospital stays and understand their associated financial impact.

# 5. Emergency admissions are significant across several conditions

# Emergency admissions represented a substantial proportion of admissions for conditions such as Heart Disease (37.95%) and Asthma (36.93%).

# Business insight:
# The relatively high emergency-admission share for some conditions may be relevant for emergency department capacity planning, staffing, and resource allocation.

# 6. Admission patterns differ by medical condition

# The percentage distribution of Elective, Emergency, Urgent, and Unknown admissions varies between medical conditions.

# Business insight:
# Healthcare managers can use condition-level admission patterns to understand where different types of care demand are concentrated and plan resources accordingly.

# 7. Data quality itself is an operational issue

# Your dataset contains "unknown" values in fields such as gender, blood type, admission type, medical condition, medication, and test result.

# Business insight:
# Incomplete patient information can reduce the reliability of operational reporting and analysis. Improving data collection and validation could improve future healthcare decision-making.

new_df.to_csv(
    r"C:\Users\bharath\OneDrive\Desktop\python\cleaned_heathcare_dataset.csv",
    index=False
)

clean_df=pd.read_csv(r"C:\Users\bharath\OneDrive\Desktop\python\cleaned_heathcare_dataset.csv")
print(clean_df)

print(clean_df.info())