#free code camp 
#1 - STATISTICS OF GIVEN LIST
#Description : Create a function named calculate() in mean_var_std.py that uses Numpy to output the 
# mean, variance, standard deviation, max, min, and sum of the rows, columns, and elements in a 3 x 3 matrix.

import numpy as np

lis=input().strip().split(" ")
lis=[int(x) for x in lis]

def calculate(lis):
    
    n=len(lis)
    if (n!=9):
        raise ValueError("List must contain 9 digits")
    
    a=np.array(lis).reshape(3,3)
    res={
        'mean':[list(np.mean(a, axis=0)), list(np.mean(a, axis=1)), np.mean(a.flatten())],
        'variance': [list(a.var(axis=0)), list(a.var(axis=1)), a.flatten().var()],
        'standard deviation': [list(np.std(a, axis=0)), list(np.std(a, axis=1)), np.std(a.flatten())],
        'max': [list(np.max(a, axis=0)), list(np.max(a, axis=1)), np.max(a.flatten())],
        'min': [list(np.min(a, axis=0)), list(np.min(a, axis=1)), np.min(a.flatten())],
        'sum': [list(np.sum(a, axis=0)), list(np.sum(a, axis=1)), np.sum(a.flatten())]
    }
    return res

print(calculate(lis))

######################################################################################################################################

#2 - DEMOGRAPHIC DATA ANALYSER 

#Description : In this challenge you must analyze demographic data using Pandas. 
# You are given a dataset of demographic data that was extracted from the 1994 Census database.
# You must use Pandas to answer the following questions:

import pandas as pd 

data=pd.read_csv("C:\\Users\\Anushree M\\Downloads\\adult.data.csv")
data.head()
data.shape
data.columns

#Q1-How many people of each race are represented in this dataset? 
#This should be a Pandas series with race names as the index labels. (race column)

print(data["race"].value_counts())

#Q2 - What is the average age of men?

avg_age_men=data.loc[data["sex"]=='Male']["age"].mean()
print(avg_age_men)

#Q3 - What is the percentage of people who have a Bachelor's degree?

sum_bachelors = data.loc[data["education"]=="Bachelors"]["education"].count()
total_education = data["education"].count()
percentage = (sum_bachelors/total_education)*100
print(percentage)

#Q4 - What percentage of people with advanced education (Bachelors, Masters, or Doctorate) make more than 50K?

edu_sal=data[["education", "salary"]]
edu_sal=edu_sal.loc[edu_sal["salary"]==">50K"]
edu_sal=edu_sal["education"]
c1=edu_sal.loc[edu_sal=="Bachelors"].count()
c2=edu_sal.loc[edu_sal=="Masters"].count()
c3=edu_sal.loc[edu_sal=="Doctorate"].count()
adv_edu_50k=c1+c2+c3
print(adv_edu_50k)

#Q5 - What percentage of people without advanced education make more than 50K?

total_edu=edu_sal.count()
total_edu

without_advedu=total_edu-adv_edu_50k
without_advedu

#Q6 - What is the minimum number of hours a person works per week?

import numpy as np 

hrs=np.array(data["hours-per-week"])
np.min(hrs)

#Q7 - What percentage of the people who work the minimum number of hours per week have a salary of more than 50K?

hrs_sal=pd.DataFrame(data[["hours-per-week", "salary"]])
total_count=hrs_sal["hours-per-week"].count()
hrs_sal=hrs_sal.loc[hrs_sal["hours-per-week"]==1]
hrs_sal=hrs_sal.loc[hrs_sal["salary"]==">50K"]
minhrs_count=hrs_sal["hours-per-week"].count()
percentage = (minhrs_count/total_count)*100
print(percentage)

#Q8 - What country has the highest percentage of people that earn >50K and what is that percentage?

country_sal=pd.DataFrame(data[["native-country", "salary"]])
country_sal=country_sal.loc[country_sal["salary"]==">50K"]["native-country"]
country_counts=country_sal.value_counts()
country_counts=pd.DataFrame(country_counts)

print(country_counts.max())
print(country_counts.max().index)

#Q9 - Identify the most popular occupation for those who earn >50K in India.

df = data.loc[data["native-country"]=='India']
df = df.loc[df["salary"]=='>50K']
df=pd.DataFrame(df["occupation"].value_counts())
print(df["occupation"].max())
print(df.index[df["occupation"]==25])


###################################################################################################################################

#3 - MEDICAL DATA VISUALIZER 

#Description : In this project, you will visualize and make calculations from medical examination data using matplotlib, seaborn, and pandas. 
# The dataset values were collected during medical examinations.
#Data description : The rows in the dataset represent patients and the columns represent information like body measurements, results from various blood tests, and lifestyle choices. 
# You will use the dataset to explore the relationship between cardiac disease, body measurements, blood markers, and lifestyle choices.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data1=pd.read_csv("C:\\Users\\Anushree M\\Downloads\\medical_examination.csv")

data1.head()
data1.shape
data1.columns
data1.info()
data1.describe()

#Add an overweight column to the data. To determine if a person is overweight, first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. 
# If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight.

subset=pd.DataFrame(data1[["weight", "height"]])

weights=list(subset["weight"])
weights=[float(x) for x in weights]

heights=list(subset["height"])
heights=[i/100 for i in heights]

heights2=[j*j for j in heights]

bmi=[x/y for x,y in zip(weights, heights2)]

subset["bmi"]=bmi

overweight = [1 if i>25 else 0 for i in bmi]

subset["overweight"]=overweight
subset["overweight"].head()
subset["overweight"].unique()

data1["overweight"] = overweight
data1["overweight"].unique()
data1["overweight"].value_counts()

#Normalize the data by making 0 always good and 1 always bad. 
# If the value of cholesterol or gluc is 1, make the value 0. If the value is more than 1, make the value 1.

chl=list(data1["cholesterol"])
glu=list(data1["gluc"])

n_cholesterol = [0 if i==1 else 1 for i in chl]
n_gluc = [0 if j==1 else 1 for j in glu]

data1["n_cholesterol"] = n_cholesterol
data1["n_gluc"] = n_gluc

data1["n_cholesterol"].head()
data1["n_cholesterol"].unique()

data1["n_gluc"].head()
data1["n_gluc"].unique()


#Convert the data into long format and create a chart that shows the value counts of the categorical features using seaborn's catplot(). 
#The dataset should be split by 'Cardio' so there is one chart for each cardio value. The chart should look like examples/Figure_1.png.

cardio_0=data1.loc[data1["cardio"]==0]
cardio_0.head()
cardio_0["cardio"].unique()

cardio_1=data1.loc[data1["cardio"]==1]
cardio_1.head()
cardio_1["cardio"].unique()

cardio_0_long = pd.melt(cardio_0, value_vars=['active', 'alco', 'n_cholesterol', 'n_gluc', 'overweight', 'smoke'])
c0l_data=pd.DataFrame(cardio_0_long.value_counts().reset_index())
c0l_data.rename(columns={0:"Count"}, inplace=True)

cardio_1_long = pd.melt(cardio_1, value_vars=['active', 'alco', 'n_cholesterol', 'n_gluc', 'overweight', 'smoke'])
c1l_data=pd.DataFrame(cardio_1_long.value_counts().reset_index())
c1l_data.rename(columns={0:"Count"}, inplace=True)

sns.barplot(data=c0l_data, x="variable", y="Count", hue="value")
plt.show()

sns.barplot(data=c1l_data, x="variable", y="Count", hue="value")
plt.show()

#Clean the data. Filter out the following patient segments that represent incorrect data:

final_data=data1
final_data.shape

#diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
final_data.drop(final_data[final_data["ap_lo"]>final_data["ap_hi"]].index, inplace=True)
final_data.shape

# height is less than the 2.5th percentile 
final_data.drop(final_data[final_data["height"] < final_data['height'].quantile(0.025)].index, inplace=True)
final_data.shape

#height is more than the 97.5th percentile
final_data.drop(final_data[final_data["height"] > final_data['height'].quantile(0.975)].index, inplace=True)
final_data.shape

# weight is less than the 2.5th percentile
final_data.drop(final_data[final_data["weight"] < final_data['weight'].quantile(0.025)].index, inplace=True)
final_data.shape

# weight is more than the 97.5th percentile
final_data.drop(final_data[final_data["weight"] > final_data['weight'].quantile(0.975)].index, inplace=True)
final_data.shape

#Create a correlation matrix using the dataset. Plot the correlation matrix using seaborn's heatmap(). 
# Mask the upper triangle. The chart should look like examples/Figure_2.png.

corr = final_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

sns.heatmap()

#########################################################################################################################################

#4 - PAGE VIEW RIME SERIES VISUALIZER

#Description : For this project you will visualize time series data using a line chart, bar chart, and box plots. 
# You will use Pandas, Matplotlib, and Seaborn to visualize a dataset containing the number of page views each day on the freeCodeCamp.org forum from 2016-05-09 to 2019-12-03. 
# The data visualizations will help you understand the patterns in visits and identify yearly and monthly growth.
# Use the data to complete the following tasks:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Use Pandas to import the data from "fcc-forum-pageviews.csv". Set the index to the date column.

tsdata = pd.read_csv("C:\\Users\\Anushree M\\Downloads\\fcc-forum-pageviews.csv", index_col='date', parse_dates=True)

tsdata.head()
tsdata.shape
tsdata.info()
tsdata.describe()

tsdata = tsdata.set_index("date")
tsdata.head()

#Clean the data by filtering out days when the page views were in the top 2.5% of the dataset or bottom 2.5% of the dataset.
#i.e. <2.5th percentile and >97.5th percentile 

drop1 = tsdata.drop(tsdata[tsdata["value"] < tsdata["value"].quantile(0.025)].index)
drop1.shape

drop2 = tsdata.drop(tsdata[tsdata["value"] > tsdata["value"].quantile(0.975)].index)
drop2.shape

#Create a draw_line_plot function that uses Matplotlib to draw a line chart similar to "examples/Figure_1.png". 
# The title should be Daily freeCodeCamp Forum Page Views 5/2016-12/2019. 
# The label on the x axis should be Date and the label on the y axis should be Page Views.

def draw_line_plot():
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(tsdata.index, tsdata["value"], 'r', linewidth=1)
    ax.set_title("Daily freeCodeCamp forum page views 5/2016-12/2019")
    ax.set_xlabel('Date')
    ax.set_ylabel('Page Views')

    #fig.savefig("fcc lineplot.png")
    return fig 

    plt.show()

draw_line_plot()

#Create a draw_bar_plot function that draws a bar chart similar to "examples/Figure_2.png". 
# It should show average daily page views for each month grouped by year. The legend should show month labels and have a title of Months. 
# On the chart, the label on the x axis should be Years and the label on the y axis should be Average Page Views.

def draw_bar_plot():
    tsdata_bar = tsdata.copy()
    tsdata_bar["Month"] = tsdata_bar.index.month
    tsdata_bar["Year"] = tsdata_bar.index.year
    tsdata_bar=tsdata_bar.groupby(["Year", "Month"])["value"].mean()
    tsdata_bar = tsdata_bar.unstack()

    fig2 = tsdata_bar.plot.bar(legend=True, xlabel="Years", ylabel="Average Page Views").figure
    plt.legend(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

    #plt.savefig("fcc barplot.png")
    return fig2

    plt.show()

draw_bar_plot()

#Create a draw_box_plot function that uses Seaborn to draw two adjacent box plots similar to "examples/Figure_3.png". 
# These box plots should show how the values are distributed within a given year or month and how it compares over time. 
# The title of the first chart should be Year-wise Box Plot (Trend) and the title of the second chart should be Month-wise Box Plot (Seasonality). 
# Make sure the month labels on bottom start at Jan and the x and y axis are labeled correctly. 
# The boilerplate includes commands to prepare the data.

def draw_box_plot():
    tsdata_box = tsdata.copy()
    tsdata_box = tsdata_box.reset_index()
    tsdata_box["Year"] = [d.year for d in tsdata_box["date"]]
    tsdata_box['month'] = [d.strftime('%b') for d in tsdata_box["date"]]

    fig, ax = plt.subplots(1, 2, figsize=(32, 10), dpi=100)

    #year-wise plot
    sns.boxplot(data = tsdata_box, x="Year", y="value", ax=ax[0])
    ax[0].set_title("Year-wise Box Plot (Trend)")
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Value")
        
    #month-wise plot
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sns.boxplot(data = tsdata_box, x="month", y="value", order=month_order, ax=ax[1])
    ax[1].title("Month-wise Box Plot (Seasonality)")
    ax[1].xlabel("Year")
    ax[1].ylabel("Value")

    #fig.savefig("fcc boxplot.png")
    return fig

    plt.show()
    

draw_box_plot()


#########################################################################################################################################

#5 - SEA LEVEL PREDICTOR

# Description : You will analyze a dataset of the global average sea level change since 1880. You will use the data to predict the sea level change through year 2050.
# Use the data to complete the following tasks:


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#Use Pandas to import the data from epa-sea-level.csv
sldata = pd.read_csv("C:\\Users\\Anushree M\\Downloads\\epa-sea-level.csv")

sldata.head()
sldata.shape
sldata.info()
sldata.describe()

#Use matplotlib to create a scatter plot using the Year column as the x-axis and the CSIRO Adjusted Sea Level column as the y-axis.

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(data=sldata, x="Year", y="CSIRO Adjusted Sea Level")
ax.set_xlabel("Year")
ax.set_ylabel("CSIRO Adjusted Sea Level")
plt.show()

#Use the linregress function from scipy.stats to get the slope and y-intercept of the line of best fit. 
# Plot the line of best fit over the top of the scatter plot. 
# Make the line go through the year 2050 to predict the sea level rise in 2050.

x=sldata["Year"]
y=sldata["CSIRO Adjusted Sea Level"]

res = stats.linregress(x,y)
slope = res.slope
yintercept = res.intercept

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(data=sldata, x="Year", y="CSIRO Adjusted Sea Level")
ax.plot(x, res.intercept + res.slope*x, 'r')
ax.set_xlim(1880, 2060)
ax.set_xlabel("Year")
ax.set_ylabel("Sea Level (inches)")
ax.set_title("Rise in Sea Level")
plt.legend()
plt.show()

#########################################################################################################################################
