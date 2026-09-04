import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df=pd.read_csv(r"C:\Users\bharath\OneDrive\Desktop\python\realistic_sales_dataset_100000_messy.csv")
sales_df=df.copy()
print(sales_df)

# inspection data
print(sales_df.info())
print(sales_df.describe())
print(sales_df.head())
print(sales_df.sample())
print(sales_df.tail())
print(sales_df.dtypes)

# duplicates handling
# count_duplicate
duplicate_count=sales_df.duplicated().sum()
print("Total duplicate rows:",duplicate_count)

# display all duplicate
duplicate_rows=sales_df[sales_df.duplicated(keep=False)]
duplicate_rows=duplicate_rows.sort_values(by="Order_ID")
print(duplicate_rows.head(10))

# shape before removing duplicate

print("before:",sales_df.shape)

# remove duplicate_rows
sales_df=sales_df.drop_duplicates()

# after removing duplicate_rows
print("after:",sales_df.shape)

# verify 
print("remaning duplicate rows",sales_df.duplicated().sum())

# null handing

print(sales_df.isnull().sum())

missing_percentage=(sales_df.isnull().sum()/len(sales_df)*100)
print(missing_percentage)

# fillna handing

sales_df["Customer_Name"]=sales_df["Customer_Name"].fillna("Unkown")

# count values
print((sales_df["Customer_Name"]=="Unkown").sum())

# number of filled values
print((sales_df["Customer_Name"]=="Unkown").sum())

# inspect a few rows
print(sales_df[sales_df["Customer_Name"]=="Unkown"].head())

print(sales_df["City"].mode())

sales_df["Payment_Method"]=sales_df["Payment_Method"].fillna("Unkown")
print(sales_df["Payment_Method"].unique())
print(sales_df.isnull().sum())


# fill missing value

sales_df["City"]=sales_df["City"].fillna(sales_df["City"].mode()[0])

print(sales_df["City"].isnull().sum())


# Display invalid rows
print(sales_df[sales_df["Quantity"] <= 0])

# Count invalid rows
print("Invalid rows:", (sales_df["Quantity"] <= 0).sum())

# Shape before cleaning
print("Before:", sales_df.shape)

# Remove invalid rows
sales_df = sales_df[sales_df["Quantity"] > 0]

# Shape after cleaning
print("After:", sales_df.shape)

# Verify
print("Remaining invalid rows:", (sales_df["Quantity"] <= 0).sum())

print(sales_df[sales_df["Customer_Rating"]<0])
print((sales_df["Customer_Rating"] < 0).sum())

print((sales_df["Customer_Rating"]<1)|(sales_df["Customer_Rating"]>5))

print("invailed rating",((sales_df["Customer_Rating"]<1)|(sales_df["Customer_Rating"]>5)).sum())

# data type

print(sales_df.dtypes)

sales_df["Order_Date"]=pd.to_datetime(
    sales_df["Order_Date"],
    errors="coerce"
)

sales_df["Ship_Date"]=pd.to_datetime(
    sales_df["Ship_Date"],
    errors="coerce"
)
print(sales_df.dtypes)

print(sales_df[["Order_Date","Ship_Date"]])


# feature enginnering

sales_df["Month"]=sales_df["Order_Date"].dt.month
sales_df["Year"]=sales_df["Order_Date"].dt.year
sales_df["Month_Name"]=sales_df["Order_Date"].dt.month_name()
sales_df["Quarter"]=sales_df["Order_Date"].dt.quarter

print(sales_df.columns)
sales_df["Profit_margin"]=(sales_df["Profit"]/sales_df["Sales"].replace(0,np.nan))*100
print(sales_df[["Sales","Profit","Profit_margin"]].head(20))

# letter formating

sales_df["Product_Name"]=(sales_df["Product_Name"].str.title().str.strip())

print(sales_df["Product_Name"].unique())

text_columns=sales_df.select_dtypes(include="str").columns
for col in text_columns:
    print("\n",col)
    print(sales_df[col].unique()[:20])
    
sales_df["City"]=(sales_df["City"].str.strip().str.title().str.replace("Chen Nai","Chennai"))
    
print(sales_df["City"].unique())

sales_df["Payment_Method"]=sales_df["Payment_Method"].str.strip().replace("",np.nan)
print(sales_df["Payment_Method"].unique())

print((sales_df["Customer_Name"]=="Unkown").sum())

print(sales_df.isnull().sum())

sales_df["Payment_Method"]=sales_df["Payment_Method"].fillna("Unkown")
print(sales_df["Payment_Method"].unique())
print(sales_df.isnull().sum())



# EDA




print(sales_df.info())

print(sales_df[sales_df["Order_Date"].isnull()][
    ["Order_ID","Order_Date","Sales","Profit"]
].head(10))

print(sales_df.describe())

print("Total_sales",sales_df["Sales"].sum())
print("Total_cost",sales_df["Cost"].sum())
print("Total_profit",sales_df["Profit"].sum())

print("Average_Sales",sales_df["Sales"].mean())
print("Average_Profit",sales_df["Profit"].mean())



# EDA-region_analysis




print(sales_df["Region"].value_counts())

Region_sales=sales_df.groupby("Region")["Sales"].sum()
print(Region_sales.sort_values(ascending=False))

Region_avg_sales=sales_df.groupby("Region")["Sales"].mean()
print(Region_avg_sales.sort_values(ascending=False))

Region_profit=sales_df.groupby("Region")["Profit"].sum()
print(Region_profit.sort_values(ascending=False))

Region_avg_Profit=sales_df.groupby("Region")["Profit"].mean()
print(Region_avg_Profit.sort_values(ascending=False))

region_summary=sales_df.groupby("Region")[["Sales","Profit"]].sum()

region_summary["Profit_Margin"]=(region_summary["Profit"]/region_summary["Sales"])*100

print(region_summary.sort_values("Profit_Margin",ascending=False))


# EDA-product_category_analysis

print("product_category",sales_df.value_counts("Product_Category").sum())

Category_sales=sales_df.groupby("Product_Category")["Sales"].sum()
print(Category_sales.sort_values(ascending=False))

Category_avg_Sales=sales_df.groupby("Product_Category")["Sales"].mean()
print(Category_avg_Sales.sort_values(ascending=False))

Category_Profit=sales_df.groupby("Product_Category")["Profit"].sum()
print(Category_Profit.sort_values(ascending=False))

Category_avg_Profit=sales_df.groupby("Product_Category")["Profit"].mean()
print(Category_avg_Profit.sort_values(ascending=False))

Category_summary=sales_df.groupby("Product_Category")[["Sales","Profit"]].sum()

Category_summary["Profit_Margin"]=(Category_summary["Profit"]/Category_summary["Sales"])*100

print(Category_summary.sort_values("Profit_Margin",ascending=False))


pivot_table=pd.pivot_table(
    sales_df,
    values="Sales",
    index="Region",
    columns="Product_Category",
    aggfunc="sum"
)
print(pivot_table)

sns.barplot(
    data=sales_df,
    x="Product_Category",
    y="Sales"
)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(
    data=pivot_table,
    annot=True,
    fmt=".0f",
    cmap="Reds",
    linewidths=1,
    linecolor="White",
    cbar=True
    
)
plt.title("sales by region and product_category")
plt.xlabel("Product_Category")
plt.ylabel("Sales")

plt.show()


print(sales_df.groupby("Product_Category")["Sales"].sum().sort_values(ascending=False))

print(sales_df.groupby("Product_Category")["Profit"].sum().sort_values(ascending=False))

print(sales_df.groupby("Product_Category")["Profit_margin"].mean())

Profit_margin=(
    sales_df.groupby("Product_Category")["Profit_margin"]
    .mean()
    .reset_index()
)

sns.barplot(
    data=Profit_margin,
    x="Product_Category",
    y="Profit_margin"
)
plt.title("Average profit_margin by product_category")
plt.xlabel("Product_Category")
plt.ylabel("Profit_margin")
plt.show()


Region_sales=sales_df.groupby("Region",as_index=False)["Sales"].sum()
print(Region_sales)

fig=px.bar(
    Region_sales,
    x="Region",
    y="Sales",
    color="Region",
    title="sales by region",
    color_discrete_map={
        "North":"lightblue",
        "South":"pink",
        "West":"lightgreen"
    }
        
    
)
fig.update_layout(
    xaxis_title="Region",
    yaxis_title="Sales",
    showlegend=True
)
fig.show()


Category_sales=sales_df.groupby("Product_Category",as_index=False)["Sales"].sum()
print(Category_sales)

fig=px.bar(
    Category_sales,
    x="Product_Category",
    y="Sales",
    title="sales by category",
    color="Product_Category",
)
fig.update_layout(
    xaxis_title="Product_Category",
    yaxis_title="Sales",
    showlegend=True
)
fig.show()

monthly_sales=sales_df.groupby(["Year","Month"],as_index=False)["Sales"].sum()
print(monthly_sales)

fig=px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    title="sales over time",
    color="Year"
)

fig.update_xaxes(
    tickmode="array",
    tickvals=list(range(1,13)),
    ticktext=[
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"
    ]
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Year",
    showlegend=True
)
fig.show()

fig=px.scatter(
    sales_df,
    x="Sales",
    y="Profit",
    color="Region",
    title="Sales vs profit by Region",
    hover_data=[
        "Product_Category",
        "Profit_margin"
    ],
    size="Quantity",
    opacity=0.3
    
)

fig.update_layout(
    xaxis_title="Sales",
    yaxis_title="Profit",
    showlegend=True
)
fig.show()

fig=px.histogram(
    sales_df,
    x="Sales",
    title="Distribution of sales",
    nbins=50
)
fig.update_layout(
    xaxis_title="Sales",
    yaxis_title="Numbers of order",
    showlegend=False
)
fig.show()

fig=px.box(
    sales_df,
    x="Region",
    y="Sales",
    color="Region",
    title="sales distribution of outlier"
)


fig.update_layout(
    xaxis_title="Region",
    yaxis_title="Sales",
    showlegend=False
)
fig.show()

Category_sales=sales_df.groupby("Product_Category",as_index=False)["Sales"].sum()
print(Category_sales)

fig = px.pie(
    Category_sales,
    names="Product_Category",
    values="Sales",
    title="Sales Share by Product Category"
)

fig.update_traces(
    textinfo="percent"
)

fig.show()



region_sales = sales_df.groupby(
    ["Region", "Product_Category"]
)["Sales"].sum()

print(region_sales.sort_values(ascending=False))



monthly_profit=sales_df.groupby(["Month","Year"],as_index=False)["Profit"].sum()
print(monthly_profit.sort_values("Profit",ascending=False))

fig=px.line(
    monthly_profit,
    x="Month",
    y="Profit",
    color="Year",
    title="Profit by months",
    markers=True
)

fig.update_xaxes(
    tickmode="array",
    tickvals=list(range(1,13)),
    ticktext=[
        "jan","feb","mar","apr","may","jun",
        "july","aug","sep","oct","nov","dec"
    ]
)


fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Profit",
    showlegend=True
)               
fig.show()                   


payment_sales=sales_df.groupby("Payment_Method")["Sales"].sum()
print(payment_sales.sort_values(ascending=False))

payment_profit=sales_df.groupby("Payment_Method")["Profit"].sum()
print(payment_profit.sort_values(ascending=False))

payment_avg_profit=sales_df.groupby("Payment_Method")["Profit"].mean()
print(payment_avg_profit.sort_values(ascending=False))

payment_profit_df=payment_profit.reset_index()

fig=px.bar(
    payment_profit_df,
    x="Payment_Method",
    y="Profit",
    title="profit by payment_method",
    text_auto=True
)
fig.update_layout(
    xaxis_title="Payment_Method",
    yaxis_title="Profit",
    showlegend=False
)
fig.show()  


print(sales_df.columns.to_list())      

customer_sales=sales_df.groupby("Customer_Name")["Sales"].sum()
print(customer_sales.sort_values(ascending=False).head(10))

unknown_count = sales_df[sales_df["Customer_Name"] == "Unknown"].shape[0]

print("Unknown customer orders:", unknown_count)

print(sales_df["Customer_Name"].unique())

unknown_customer = sales_df[sales_df["Customer_Name"] == "Unkown"]

print("Unkown customer orders:", unknown_customer.shape[0])
print("Unkown customer sales:", unknown_customer["Sales"].sum())

customer_df=sales_df[sales_df["Customer_Name"]!="Unknow"]

customer_sales=customer_df.groupby("Customer_Name")["Sales"].sum()
print(customer_sales.sort_values(ascending=False).head(10))

print(sales_df["Customer_Name"].value_counts().head())
print(sales_df["Customer_Name"].unique()[:10])

customer_df = sales_df[sales_df["Customer_Name"] != "Unkown"]

print((customer_df["Customer_Name"] == "Unkown").sum())

customer_sales = (
    customer_df.groupby("Customer_Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(customer_sales)

top10_customer_df = (
    customer_sales
    .head(10)
    .sort_values()
    .reset_index()
)

print(top10_customer_df)

fig=px.bar(
    top10_customer_df,
    x="Sales",
    y="Customer_Name",
    orientation="h",
    title="top 10 customer by sales"
)

fig.update_layout(
    xaxis_title="Customer_Name",
    yaxis_title="Sales",
    showlegend=False
)
fig.show()

top10_sales=customer_sales.head(10).sum()
total_customer_sales=customer_df["Sales"].sum()

top10_percentage=(top10_sales/total_customer_sales)*100

print("Top 10 customer sales:", top10_sales)
print("Top 10 customer contribution:", top10_percentage)
print("Top 10 customer contribution:", top10_percentage, "%")


customer_df=sales_df[sales_df['Customer_Name']!="Unkown"]


customer_profit=(
    customer_df.groupby("Customer_Name")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

top10_profit=customer_profit.head(10)
print(top10_profit)



plt.figure(figsize=(10,6))


top10_profit_df=(
    top10_profit
    .sort_values()
    .reset_index()
)

sns.barplot(
    data=top10_profit_df,
    x="Profit",
    y="Customer_Name"
)
plt.title("profit_top10_customer")
plt.xlabel("Profit")
plt.ylabel("Customer_Name")
plt.show()


top10_profit_value = top10_profit.sum()

total_profit = customer_df["Profit"].sum()

top10_profit_percentage = (
    top10_profit_value / total_profit
) * 100

print("Top 10 customer profit:", top10_profit_value)
print("Top 10 profit contribution:", top10_profit_percentage, "%")
    
        
        
# # ==========================================
# # BUSINESS INSIGHTS
# # ==========================================

# 1. Regional insight
# South region generated the highest total sales and profit, making it the strongest-performing region.

# 2. Category insight
# Office had the highest total sales, while Electronics generated the highest total profit.

# 3. Customer insight
# The top 10 customers contributed only about 3.21% of total customer sales, indicating that sales are not heavily dependent on a small group of customers.

# 4. Profit customer insight
# The top 10 customers contributed about 2.73% of total customer profit, also suggesting relatively broad profit distribution.

# 5. Business implication
# The company should continue strengthening the South region while investigating what factors are driving its stronger performance and whether similar strategies can be applied to North and West. 



sales_df.to_csv(
    r"C:\Users\bharath\OneDrive\Desktop\python\sales_cleaned.csv",
    index=False
)


clean_df = pd.read_csv(
    r"C:\Users\bharath\OneDrive\Desktop\python\sales_cleaned.csv"
)

print(clean_df.shape)
print(clean_df.head())
print(clean_df.isnull().sum())
print(clean_df.dtypes)