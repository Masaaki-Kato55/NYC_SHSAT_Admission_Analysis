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
middleSchoolDataLabeled = pd.read_csv('data/middleSchoolData.csv')

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

#%% Question 4: Is there a relationship between how students perceive
# their school (as reported in columns L-Q) and how the school performs on
#objective measures of achievement (as noted in columns V-X).

# Brainstorm: so we are looking for relationships. So maybe correlation?
# But, there are too many variables
# maybe doing a PCA first
# doing a PCA for each group and finding one factor that capture same information

#%% Question 4A: HOw students perceive their school
schoolClimate = unlabeledArray[:,9:15]
#looking at correlation
schoolClimateCorr = r = np.corrcoef(schoolClimate,rowvar=False)
# lets look at raw data
plt.imshow(r) # Display an image, i.e. data, on a 2D regular raster.
plt.colorbar()

#%% Question 4A: PCA
# 1. Z-score the data:
zscoredData = stats.zscore(schoolClimate)

# 2. Run the PCA:
pca = PCA().fit(zscoredData)
# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_
# 3b. Loadings (eigenvectors):
loadings = pca.components_
# 3c. Rotated Data:
rotatedData = pca.fit_transform(zscoredData)
# 4. For the purposes of this,
covarExplained = eigVals/sum(eigVals)*100

#%% Question 4A: Screeplot
numClasses = 6
plt.bar(np.linspace(1,6,6),eigVals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('PCA on column L-Q')
plt.plot([0,numClasses],[1,1],color='red',linewidth=1)

# PCA 1 explains most of the variance

#%% Question 4A:
whichPrincipalComponent = 0 # Try a few possibilities (at least 1,2,3 - or 0,1,2 that is - indexing from 0)

# 1: The first one accounts for almost everything, so it will probably point
# in all directions at once
# 2: Challenging/informative - how much information?
# 3: Organization/clarity: Pointing to 6 and 5, and away from 16 - structure?

plt.bar(np.linspace(1,6,6),loadings[whichPrincipalComponent,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')

# question 1: Rigorous Instructions
schoolClimatePCA = rotatedData[:,0]*-1

#%% Question 4B:how the school performs on
#objective measures of achievement (as noted in columns V-X).
schoolAchievement = unlabeledArray[:,19:22]

schoolAchievementCorr = np.corrcoef(schoolAchievement,rowvar=False)
# lets look at raw data
plt.imshow(schoolAchievementCorr) # Display an image, i.e. data, on a 2D regular raster.
plt.colorbar()

# Seems like there is one big cluster and one small cluster

#%% Question 4B: PCA
# 1. Z-score the data:
zscoredData1 = stats.zscore(schoolAchievement)

# 2. Run the PCA:
pca1 = PCA().fit(zscoredData1)
# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals1 = pca1.explained_variance_
# 3b. Loadings (eigenvectors):
loadings1 = pca1.components_
# 3c. Rotated Data:
rotatedData1 = pca1.fit_transform(zscoredData1)
# 4. For the purposes of this,
covarExplained1 = eigVals1/sum(eigVals1)*100

# PCA 1 explains over 75% of variance

#%% Question 4B: Screeplot
numClasses1 = 3
plt.bar(np.linspace(1,3,3),eigVals1)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('PCA on column V-X')
plt.plot([0,numClasses1],[1,1],color='red',linewidth=1)

# Based on elbow method, we only include PCA 1

#%% Question 4B: Loading Interpretation

whichPrincipalComponent = 0 # Try a few possibilities (at least 1,2,3 - or 0,1,2 that is - indexing from 0)

# 1: The first one accounts for almost everything, so it will probably point
# in all directions at once
# 2: Challenging/informative - how much information?
# 3: Organization/clarity: Pointing to 6 and 5, and away from 16 - structure?

plt.bar(np.linspace(1,3,3),loadings1[whichPrincipalComponent,:])
plt.xlabel('Question')
plt.ylabel('Loading')

# question 1: school achievement
schoolAchievementPCA = rotatedData1[:,0]

#%% Question 4C: Correlation between school achievement and school climate
climateAchievementCorr, climateAchievementPVal = pearsonr(schoolClimatePCA, schoolAchievementPCA)
print(climateAchievementPVal) # significant
plt.scatter(schoolClimatePCA, schoolAchievementPCA)
plt.xlabel('schoolClimatePCA')
plt.ylabel('schoolAchievementPCA')
plt.title('Correlation (r=0.4)')
plt.show()
# 0.4 correlation
# makes sense, the more rigoruous the institution is,
# the better school achievemenst are

#%% Question 7: What proportion of schools account for all students
# accepted to HSPHS

# number of acceptances
acceptanceOver0 = rowWiseData[rowWiseData['acceptances']>0]

#print(acceptanceOver0.shape) # 291 rows

acceptanceOver1 = rowWiseData[rowWiseData['acceptances']>1] # 212 rows
print(212/291)
# x/291 = .9, x = .9*291
numSchools = .9*291
print(numSchools)
print(262/291)
acceptance1 = rowWiseData[rowWiseData['acceptances']==1] #79

# 262 schools account for 90% of students accepted to HSPHS
# proportion wise that is 262/449
print(262/449) #.58

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
temp2 = unlabeledArray[:,2:19]
predictors = np.column_stack((temp1,temp2)) # index 0 - applicants

#%% Correlation Matrix:

r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r)
plt.colorbar()

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