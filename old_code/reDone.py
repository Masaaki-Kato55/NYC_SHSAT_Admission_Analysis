#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 22:07:53 2021
ALOHA
@author: Masaaki Kato
"""
#%% Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn.decomposition import PCA
import seaborn as sns
#%% Import Data
middleSchoolDataLabeled = pd.read_csv('../data/middleSchoolData.csv')

#%% Data Cleaning
# row-wise removal
rowWiseData = middleSchoolDataLabeled.dropna()

# Roughly lost 1/4 of our data. All of the charter school data
# Our analysis only pertains to NYC middle schools that are public schools
#%% Data Labeling
unlabeledData = rowWiseData.iloc[:,2:]
unlabeledArray = unlabeledData.to_numpy()

#%% PCR Model
# Steps:
# 1: split dataset. features and outcomes (admission, objective acheivements)
# 2: PCA
# 3: Multiple Linear Regression

#%% Question 8A: data formatting
# outcome1: admission
# outcome2: objective acheivements
# features: all else

outcome1 = stats.zscore(unlabeledArray[:,1]) # admission
outcome2 = stats.zscore(unlabeledArray[:,19]) # objective achievements

# features:
temp1 = unlabeledArray[:,0] # applicants
temp2 = unlabeledArray[:,2:19] # rest of features
predictors = np.column_stack((temp1,temp2)) # index 0 - applicants

#%% Correlation Matrix:

r = np.corrcoef(predictors,rowvar=False)
sns.set(rc={'figure.figsize':(15,12)})
sns.heatmap(data=r, annot=True,cmap="vlag") # "icefire"
#plt.imshow(r)
#plt.colorbar()
# variables 8-13 are highly correlated

#%% Initial Multiple Regression

#%% PCA
# Z-score data & run PCA again:
zscoredData2 = stats.zscore(predictors)
pca3 = PCA().fit(zscoredData2)
eigValues3 = pca3.explained_variance_
loadings3 = pca3.components_
origDataNewCoordinates = pca3.fit_transform(zscoredData2)
covarExplained3 = eigValues3/sum(eigValues3)*100
# Scree plot:
numPredictors = 18
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues3)
plt.plot([1,numPredictors],[1,1],color='red') # Kaiser criterion line
#plt.xticks(1,18)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')

# Based on Kaiser criterion: looking at Eigensum of 1 or greater
# PCA 1-4

#%% Looking at the corrected scree plot, we get 2 factors, both by
# Kaiser criterion and Elbow

# look at meaning
whichPrincipalComponent = 1 #

plt.bar(np.linspace(1,18,18),loadings3[whichPrincipalComponent,:])
plt.xlabel('Factor')
plt.ylabel('Loading')

# Column names:
name_dict = {'application':1,'spending':2, 'class_size':3,
             'asian':4,'black':5,'white':6,'hispanic':7,
             'multi_race':8,'rigorous_instructions':9,
             'collaborative_teachers':10,'supportive_environment':11,
             'effective_school_leadership':12,'family_community_tie':13,
             'trust':14,'disability':15,'poverty':16,'ESL':17,'school_size':18}

# PCA 1
# looking at factor 1,3,4,7-14
# Positive School Climate

# PCA 2
# factor 1,3-,4,7-,8-,18
# Large population

# PCA 3
# factor 5,16-18
# improvished coomunities

# PCA 4
# factor 2,6-8,13,17
# well-offness

X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],
                           origDataNewCoordinates[:,2],origDataNewCoordinates[:,3]]))

#%% Question 8B: Looking at correlation of X

# show these variables are uncorrelated with each other
a1 = np.corrcoef(X,rowvar=False)
# lets look at raw data
plt.imshow(a1) # Display an image, i.e. data, on a 2D regular raster.
plt.colorbar()

#%% Question 8B: Multiple Linear Regression,
X1 = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],
                            origDataNewCoordinates[:,2],origDataNewCoordinates[:,3]]))
Y1 = stats.zscore(unlabeledArray[:,1]) # admission

# for admission
regr1 = linear_model.LinearRegression()
regr1.fit(X1,Y1)
rSqr1 = regr1.score(X1,Y1) # variance explained
betas1 = regr1.coef_ # m
yInt = regr1.intercept_  # b

print(rSqr1,
      betas1)

# Results:
# r^2 = 0.45
# Betas:
# PCA1 (Positive School Climate): 0.25
# PCA2 (Large Population): 0.17
# PCA3 (Improvished communities): 0.04
# PCA4 (financial stability): -0.08

# visualizing results as tables
from tabulate import tabulate



#%% Question 8C: Multiple Linear Regression for objective achievements
X1 = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],
                            origDataNewCoordinates[:,2],origDataNewCoordinates[:,3]]))
Y2 = stats.zscore(unlabeledArray[:,19])

# regress!!
regr2 = linear_model.LinearRegression()
regr2.fit(X1,Y2)
rSqr2= regr2.score(X1,Y2) # variance explained
betas2 = regr2.coef_ # m
yInt2 = regr2.intercept_  # b

print(rSqr2,
      betas2)

# Results:
# r^2: 0.25
# PCA1 (positive school climate): 0.20
# PCA2: -0.10
# PCA3: 0.06
# PCA4: 0.09

from tabulate import tabulate


#%% Visualizing results

y_hat1 = betas1[0]*origDataNewCoordinates[:,0] + betas1[1]*origDataNewCoordinates[:,1] + betas1[2]*origDataNewCoordinates[:,2] + betas1[3]*origDataNewCoordinates[:,3]+ yInt
plt.plot(y_hat1,Y1,'o',markersize=.75) # y_hat, income
plt.xlabel('Prediction from model')
plt.ylabel('Actual acceptances')
plt.title('R^2: {:.3f}'.format(rSqr1))



#%% Visualizing Results 2
y_hat2 = betas2[0]*origDataNewCoordinates[:,0] + betas2[1]*origDataNewCoordinates[:,1] + betas2[2]*origDataNewCoordinates[:,2] + betas2[3]*origDataNewCoordinates[:,3]+ yInt2
plt.plot(y_hat2,Y2,'o',markersize=.75) # y_hat, income
plt.xlabel('Prediction from model')
plt.ylabel('Actual achievement scores')
plt.title('R^2: {:.3f}'.format(rSqr2))