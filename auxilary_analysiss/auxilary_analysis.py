#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 22:07:53 2021
ALOHA
@author: mkpanda
"""
#%% Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn.decomposition import PCA

from scipy.stats import levene


#%% Import Data
middleSchoolDataLabeled = pd.read_csv('../data/middleSchoolData.csv')

#%% Data Cleaning
# row-wise removal
rowWiseData = middleSchoolDataLabeled.dropna()
#print(middleSchoolDataLabeled.info())
#print(rowWiseData.info())

# Roughly lost 1/4 of our data. All of the charter school data
# Our analysis only pertains to NYC middle schools that are public schools
#%% Data Labeling
unlabeledData = rowWiseData.iloc[:,2:]
unlabeledArray = unlabeledData.to_numpy()


#%% Quesetion 5: Test a hypothesis of your choice as to which kind of school
# (e.g. small schools vs. large schools or charter schools vs. not
# (or any other classification, such as rich vs. poor school)) performs
# differently than another kind either on some dependent measure, e.g.
# objective measures of achievement or admission to HSPHS (pick one).


# Richer schools have better academic outcomes?
# H0: school achievements of rich and poor schools are the sames
# H1: schools achievmenets are different for rich and poor schools
# define rich as schools, that higher per student spending than the median

schoolSpending = unlabeledArray[:,2]
spendingMedian = np.median(schoolSpending)

# median is 20096
# split data based on poor/rich
richSchools = unlabeledArray[unlabeledArray[:,2]>spendingMedian]
poorSchools = unlabeledArray[unlabeledArray[:,2]<=spendingMedian]
richSchoolsAch = richSchools[:,19]
poorSchoolsAch = poorSchools[:,19]
richSchoolAccept = richSchools[:,1]
poorSchoolAccept = poorSchools[:,1]

# Conducting t-test
#%% Question 5: Checking HOV
stats1,p = levene(richSchoolsAch,poorSchoolsAch)
t1,p1 = stats.ttest_ind(poorSchoolsAch, richSchoolsAch)
print(t1,p1)

# there is a significant
# levene test, not significant. so assume HOV
#%% Question 6: t-test for acceptances
stats2,p2 = levene(richSchoolAccept,poorSchoolAccept)
t3,p3 = stats.ttest_ind(richSchoolAccept,poorSchoolAccept)
print(stats2, p2)
print(t3,p3)

richMean = np.mean(richSchoolAccept)
poormean = np.mean(poorSchoolAccept)
richStd = np.std(richSchoolAccept)
poorStd = np.std(poorSchoolAccept)
print(richMean, poormean)
print(richStd,poorStd)

plt.plot(richSchoolAccept)
plt.plot(poorSchoolAccept)
#%% Question 6: Is there any evidence that the availability of material
# resources (e.g. per student spending or class size) impacts objective
# measures of achievement or admission to HSPHS?
# looking at correlation (one for each)
# then looking at ANOVA

# assign variables we need: class size, per student spending,
# objective acheivements, HSPHS admission
admission = unlabeledArray[:,1]
classSize = unlabeledArray[:,3]
schoolSpending = unlabeledArray[:,2]
objectiveAch = unlabeledArray[:,19]

# Reasoning of variables: concering objective acheivement, in our PCA results,
# column 19 was the explained the most variance (65%). So chose that.

# Brainstorm: How to measure material resources
# comparing different resources (money, class size)
print(pearsonr(classSize, schoolSpending)) # -0.46
plt.scatter(classSize, schoolSpending)
plt.xlabel('classSize')
plt.ylabel('school_Spending')
plt.title('Correlation (r=-0.46)')

# Observation: as class size increases, there is less funding per student

#%% Question 6A: Correlation

# in terms of achievement

# looking at class size
print(pearsonr(classSize, objectiveAch)) # 0.21
# little connection overall

# looking at per student spending
print(pearsonr(schoolSpending, objectiveAch)) # -0.15
#plt.scatter(schoolSpending, objectiveAch)
#plt.xlabel('school_spending')
#plt.ylabel('objectiveAch')
# the more school spendings increase, the lower the score?
# higher proportion of schools with lower spending and scores are spread out.
# teaching styles?

# in terms of admission to HSPHS

# looking at class size
print(pearsonr(classSize, admission)) # 0.356
plt.scatter(classSize, admission) #
# schools with larger class size have higher admissions

print(pearsonr(schoolSpending,admission)) # -0.34
plt.scatter(schoolSpending, admission)
# more spending does not mean, more admission

fig, axs = plt.subplots(2, 2)
axs[0,0].scatter(schoolSpending, objectiveAch)
axs[1,0].scatter(schoolSpending, admission)
axs[0,1].scatter(classSize, objectiveAch)
axs[1,1].scatter(classSize, admission)
axs[1,1].set(xlabel='class_size')
axs[1,0].set(xlabel='school_spending')
axs[0,0].set(ylabel='achievement_score')
axs[1,0].set(ylabel='acceptances')

#%% Question 6B: Multiple Linear Regression

# since doing a predictor, we are conducting a multiple regression
X = np.vstack((classSize,schoolSpending)).T
Y = objectiveAch
regr = linear_model.LinearRegression()
regr.fit(X,Y) # use fit method
rSqr = regr.score(X,Y) #
betas = regr.coef_ #m

# conclusion:
# objective achievement: r^2 = 0.05 poor predictor
# admission: r^2 = 0.17. poor predictor

#%% Question 6C: Conclusion
# The availability of material resources does not impact future acheivements (admission, scores)
