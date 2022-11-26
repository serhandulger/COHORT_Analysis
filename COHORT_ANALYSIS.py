import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

import holoviews as hv
from holoviews import opts
%matplotlib inline
import altair as alt
import plotly.express as px
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

df = pd.read_csv('/Users/serhandulger/online.csv')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### NA SUM #####################")
    print(dataframe.isnull().sum().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())
    print("##################### Nunique #####################")
    print(dataframe.nunique())

df = df.iloc[:,1:9]

check_df(df)

import datetime as dt
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")

cat_cols = [col for col in df.columns if df[col].dtype == 'O']
num_cols = [col for col in df.columns if df[col].dtype != 'O' and df[col].dtype != 'datetime64' ]

cat_cols
num_cols

# EXPLORATORY DATA ANALYSIS
order_percentage = df.groupby(["CustomerID"])["InvoiceNo"].nunique()
order_percentage.sort_values(ascending=False)

import numpy as np
mult_orders_perc = np.sum(order_percentage > 1) / df['CustomerID'].nunique()
print(f'{100 * mult_orders_perc:.2f}% of customers ordered more than once.')

import seaborn as sns
ax = sns.distplot(order_percentage, kde=False, hist=True)
ax.set(title='Distribution of number of orders per customer',
       xlabel='# of orders',
       ylabel='# of customers');

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

eda1 = df.groupby(["Description"])["Quantity"].sum().reset_index().sort_values(by="Quantity",ascending=False).head(10)

eda1.rename(columns={'Description':'Product_name'},inplace=True)

# top 10 products by quantity
plt.figure(figsize=(12,6))
sns.barplot(x=eda1['Quantity'],y=eda1['Product_name'])
plt.title('Top 10 products by quantity')

df["year"] = df["InvoiceDate"].apply(lambda x: x.year)
df["month"] = df["InvoiceDate"].apply(lambda x: x.month)
df["hour"] = df["InvoiceDate"].apply(lambda x: x.hour)
df["day"] = df["InvoiceDate"].apply(lambda x: x.day)

df.hour.unique()

def timing_of_day(time):
    if(time==6 or time==7 or time==8 or time==9 or time==10 or time==11):
        return "MORNING"
    elif (time==12 or time==13 or time==14 or time==15 or time==16 or time==17):
        return "AFTERNOON"
    else:
        return "EVENING"

df['day_time_part'] = df['hour'].apply(timing_of_day)

df.head(2)

sales_time_patterns = df.groupby('day_time_part')['TotalAmount'].sum().reset_index().sort_values('TotalAmount',ascending=False)
sales_time_patterns

plt.figure(figsize=(12,6))
sns.barplot(x=sales_time_patterns['day_time_part'],y=sales_time_patterns['TotalAmount'])
plt.title('Sales count in different day timings')

df['month_name'] = df['InvoiceDate'].dt.month_name()

# extracting day from the Invoice date

df['Day']=df['InvoiceDate'].dt.day_name()

month_patterns = df.groupby('month_name')['TotalAmount'].sum().reset_index().sort_values('TotalAmount',ascending=False)

month_patterns

# Sales volume by months
plt.figure(figsize=(20,6))
sns.barplot(x=month_patterns['month_name'],y=month_patterns['TotalAmount'])
plt.title('Sales in different Months ')

df.year.value_counts()

# Number of items sold by years
df1 = df.groupby(['year']).filter(lambda x: (x['year'] == 2010).any())
df2 = df.groupby(['year']).filter(lambda x: (x['year'] == 2011).any())

sales_2010=hv.Bars(df1.groupby(['month'])['Description'].count()).opts(ylabel="# of items", title='# of items sold in 2010')
sales_2011=hv.Bars(df2.groupby(['month'])['Description'].count()).opts(ylabel="# of items", title='# of items sold in 2011')

(sales_2010 + sales_2011).opts(opts.Bars(width=380, height=300,tools=['hover'],show_grid=True))

# Number of unique items sold by years
df1 = df.groupby(['year']).filter(lambda x: (x['year'] == 2010).any())
df2 = df.groupby(['year']).filter(lambda x: (x['year'] == 2011).any())

sales_2010=hv.Bars(df1.groupby(['month'])['Description'].nunique()).opts(ylabel="# of items", title='# of unique items sold in 2010')
sales_2011=hv.Bars(df2.groupby(['month'])['Description'].nunique()).opts(ylabel="# of items", title='# of unique items sold in 2011')

(sales_2010 + sales_2011).opts(opts.Bars(width=380, height=300,tools=['hover'],show_grid=True))

#Plotting day transaction across a typical month in 2010 and 2011
sales_day=hv.Curve(df.groupby(['day'])['Description'].count()).opts(ylabel="# of items", title='Cummulative day transactions-2010 & 2011')

#Line chart
sales_day.opts(opts.Curve(width=800, height=300,tools=['hover'],show_grid=True))

#Setting plot style
plt.figure(figsize = (15, 8))
plt.style.use('seaborn-white')

#Top 10 fast moving products
plt.subplot(1,2,1)
ax=sns.countplot(y="Description", hue="year", data=df, palette="pastel",
              order=df.Description.value_counts().iloc[:10].index)

ax.set_xticklabels(ax.get_xticklabels(),fontsize=11,rotation=40, ha="right")
ax.set_title('Top 10 Fast moving products',fontsize= 22)
ax.set_xlabel('Total # of items purchased',fontsize = 20)
ax.set_ylabel('Top 10 items', fontsize = 20)
plt.tight_layout()

#Bottom 10 fast moving products
plt.subplot(1,2,2)
ax=sns.countplot(y="Description", hue="year", data=df, palette="pastel",
              order=df.Description.value_counts().iloc[-10:].index)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=11,rotation=40, ha="right")
ax.set_title('Bottom 10 Fast moving products',fontsize= 22)
ax.set_xlabel('Total # of items purchased',fontsize = 20)
ax.set_ylabel('Bottom 10 items', fontsize = 20)
plt.tight_layout()

# DATA STANDARDIZATION

def get_month(x):
    return dt.datetime(x.year,x.month,1)

df["InvoiceMonth"] = df["InvoiceDate"].apply(get_month)

df.head(2)
df.shape

grouping = df.groupby(["CustomerID"])["InvoiceMonth"]
grouping.head()

df["CohortMonth"] = grouping.transform("min")

df.head()
df.shape

def get_date(df,column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year,month,day

#Assign time offset value

# Getting the integers for date parts from the InvoiceDaycolumn

invoice_year, invoice_month, _ = get_date(df,"InvoiceMonth")

# Getting the integers for date parts from the CohortDay column

cohort_year, cohort_month, _ = get_date(df,"CohortMonth")

# Calculating difference in years

years_diff = invoice_year - cohort_year

# Calculating difference in months

months_diff = invoice_month - cohort_month

# Calculating difference in days

#days_diff = invoice_day - cohort_day
df["CohortIndex"] = years_diff*12+months_diff+ 1
df.head()

df.shape

cohort_data = df.groupby(["CohortMonth","CohortIndex"])["CustomerID"].nunique()
cohort_data

cohort_data = cohort_data.reset_index()
cohort_data.head()

cohort_counts = cohort_data.pivot(index="CohortMonth",
                                 columns="CohortIndex",
                                 values="CustomerID")

cohort_counts

cohort_sizes = cohort_counts.iloc[:,0]
cohort_sizes

retention = cohort_counts.divide(cohort_sizes,axis=0)
retention

retention.round(3)*100

# Heatmap of Retention TABLE

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.title("Retention Rates")
sns.heatmap(data=retention,
           annot=True,
           fmt=".0%",
           vmin = "0.0",
           cmap = "BuGn")
plt.show()

# Calculating Average Price COHORT Tables

## We will be analyzing the average price metric and analyze if there are any differences in shopping patterns across time and cohorts.

cohort_data = df.groupby(["CohortMonth","CohortIndex"])["UnitPrice"].mean()
cohort_data = cohort_data.reset_index()
cohort_data

average_price = cohort_data.pivot(index="CohortMonth",
                                 columns="CohortIndex",
                                 values="UnitPrice")

print(average_price.round(1))

import seaborn as sns
plt.figure(figsize=(8, 6))
plt.title('Average Spend by Monthly Cohorts')
sns.heatmap(average_price, annot=True, cmap='Blues')
plt.show()

# FUNCTIONALIZE ALL PROCESS

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

df = pd.read_csv('/Users/serhandulger/online.csv')
df = df.iloc[:,1:9]

import datetime as dt
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

def get_month(x):
    return dt.datetime(x.year,x.month,1)

def get_date(df,column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year,month,day


def calculate_cohort(df):
    df["InvoiceMonth"] = df["InvoiceDate"].apply(get_month)

    grouping = df.groupby(["CustomerID"])["InvoiceMonth"]

    df["CohortMonth"] = grouping.transform("min")

    invoice_year, invoice_month, _ = get_date(df, "InvoiceMonth")

    cohort_year, cohort_month, _ = get_date(df, "CohortMonth")

    # Calculate difference in years

    years_diff = invoice_year - cohort_year

    # Calculate difference in months

    months_diff = invoice_month - cohort_month

    df["CohortIndex"] = years_diff * 12 + months_diff + 1
    cohort_data = df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique()
    cohort_data = cohort_data.reset_index()
    cohort_counts = cohort_data.pivot(index="CohortMonth",
                                      columns="CohortIndex",
                                      values="CustomerID")

    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)
    retention_rate = retention.round(3) * 100
    return cohort_sizes, retention, retention_rate

cohort_sizes,retention,retention_rate = calculate_cohort(df)

with sns.axes_style("white"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
    import matplotlib.colors as mcolors
    # retention matrix
    sns.heatmap(retention_rate,
                mask=retention_rate.isnull(),
                annot=True,
                fmt='g',
                cmap='Greens',
                vmin=0.0,
                vmax=50,
                ax=ax[1])
    ax[1].set_title('Customer Retention Rate by Monthly Cohorts', fontsize=16)
    ax[1].set(xlabel='Cohort Index',
              ylabel='')

    # cohort size
    cohort_size_df = pd.DataFrame(cohort_sizes).rename(columns={1: 'Cohort Size'})
    white_cmap = mcolors.ListedColormap(['white'])
    sns.heatmap(cohort_size_df,
                annot=True,
                cbar=False,
                fmt='g',
                cmap=white_cmap,
                ax=ax[0])

    fig.tight_layout()
    plt.show()