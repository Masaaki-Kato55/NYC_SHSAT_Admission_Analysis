#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 22:07:53 2021

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
middleSchoolDataLabeled = pd.read_csv('middleSchoolData.csv')

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

#%% Question 1: Correlation between number of applications and admissions to HSPHS
# import pearsonr

application = unlabeledArray[:,0]
admission = unlabeledArray[:,1]

# computing correaltion coefficient
appAcceptCorr, appAcceptPVal = pearsonr(application, admission)

print(appAcceptPVal) # significant
# plot scatterplot
plt.scatter(application,admission)
plt.xlabel('application')
plt.ylabel('admission')
plt.title('Correlation (r=0.81)')

# The correlation between the number of applications and admissions to HSPHS is: 0.80527

#%% Question 2: What is a better predictor of admission to HSPHS? 
# Raw number of applications or application rate?

rawApplication = unlabeledArray[:,0]
schoolSize = unlabeledArray[:,18]
applicationRate = rawApplication / schoolSize

# since doing a predictor, we are conducting a multiple regression
X = np.vstack((rawApplication, applicationRate)).T # raw number of apps, app rates
Y = admission # admissions
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) #
betas = regr.coef_ #m

print(betas)

# betas: 
    # number of raw applications: 0.24431
    # application rates: 62.38054'

#%% Question 2: Regression for each
# for class size
data1 = np.vstack((rawApplication, admission)).T
from simple_linear_regress_func_copy import simple_linear_regress_func
output1 = simple_linear_regress_func(data1) #(m,b,r^2)
print('applicants on admission:',output1)
# for application rate 
data2 = np.vstack((applicationRate, admission)).T
output2 = simple_linear_regress_func(data2)
print('applicationRate on admission:', output2)

#%% Question 3: Which school has the best *per student* odds of sending 
# someone to HSPHS?

# brainstorm: if I apply to HPSHS, what are my odds of getting 
# p(accepted|applied)
# create this metric as a new column in df (labeled data)
dataq3 = rowWiseData.copy() 

# Steps:
    # 1: calculating probability of applying 
        # num of applications / school size
    # 2: calculating probability of being accepted given applied
        # p(accepted|applied) = p(accept AND applied) / p(applied)
        
    # 3: using that as new column

probApply = dataq3['applications'] / dataq3['school_size'] 
probApplyAndAccepted = dataq3['acceptances'] / dataq3['school_size']
condAcceptedGivenApply = probApplyAndAccepted / probApply
dataq3['condAcceptedGivenApplied'] = condAcceptedGivenApply

sorted3 = dataq3.sort_values('condAcceptedGivenApplied', ascending=False)

# answer: THE CHRISTA MCAULIFFE SCHOOL\I.S. 187

#%% Question 3B:
dataq3['acceptance_rate'] = dataq3['acceptances'] / dataq3['applications']
sorted4 = dataq3.sort_values('acceptance_rate', ascending=False)



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
# This makes sense... (Stuy??)

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

#%% Question 8: Build a model of your choice – clustering, classification or prediction – 
#that includes all factors – as to what school characteristics are most important in terms of 
# a) sending students to HSPHS, b) achieving high scores on objective measures of achievement?

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

























