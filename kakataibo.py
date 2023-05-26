# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:31:33 2023

@author: jveraz
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

datos = pd.read_excel('kakataibo_datos.xlsx')
print(datos.columns)

#############################
## multivariate regression ##
#############################

## data
LL = []
for i in datos.index:
    L = list(datos.iloc[i])
    LL += [L]

X = []
y = []
for L in LL:
    X += [L[1:]]
    y += [L[0]]

## train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

## multivariate regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
I = dict(zip(datos.columns[1:],regressor.feature_importances_))

## plot importances
keys = I.keys()
values = I.values()

fig, ax = plt.subplots(dpi=800)
plt.bar(keys, values)
labels = ['SR', 'SR tail-head','nominalizations', 'words', 'sentences']
ax.set_xticklabels(labels)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.ylabel(r'importance for RF prediction',fontsize=12)
plt.xlabel(r'variable',fontsize=15)
plt.rcParams.update({'font.size': 12})
plt.savefig('I.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=800)
plt.show()
## prediction error
predictions = regressor.predict(X_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2))

## mape
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

pred = regressor.predict(X)

## correlation between age and predicted age
corr, _ = spearmanr(np.array(y), np.array(pred))
print('Spearmans correlation: %.3f' % corr)

## plot
fig, ax = plt.subplots(dpi=800)
ax.plot(y,pred,marker='o',color='orange',linewidth=0,markersize=5,markeredgewidth=0.5,markeredgecolor=None,alpha=0.85,fillstyle='full',clip_on=True)
plt.ylabel(r'predicted age',fontsize=15)
plt.xlabel(r'age',fontsize=15)
plt.rcParams.update({'font.size': 12})
plt.savefig('pred.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=800)
plt.show()

#########################
## linear regressions! ##
#########################

## SR versus age
lin = LinearRegression()
lin.fit(datos[['age']], datos['Switch-reference'])
b = np.round(lin.intercept_,2)
m = np.round(lin.coef_[0],2)
r2 = np.round(lin.score(datos[['age']], datos['Switch-reference']),2)
print(lin.summary())
fig, ax = plt.subplots(dpi=800)
ax.plot(datos[['age']], datos['Switch-reference'],marker='o',color='orange',linewidth=0,markersize=5,markeredgewidth=0.25,markeredgecolor='k',alpha=0.85,fillstyle='full',clip_on=True)
ax.plot(datos['age'], lin.predict(datos[['age']]),color='k',linewidth=1.75,markersize=0,markeredgewidth=0.5,markeredgecolor='k',alpha=0.75,fillstyle='full',label=r'$R^2 = $'+str(r2))
plt.grid(False)
plt.legend(loc='best',fontsize=7)
plt.ylabel(r'number of switch reference markers',fontsize=12)
plt.xlabel(r'age',fontsize=12)
plt.rcParams.update({'font.size': 10})
plt.savefig('SRvsage.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=1024)
plt.show()

## SR tail head versus age
lin = LinearRegression()
lin.fit(datos[['age']], datos['Switch-reference-markers-tail-head'])
b = np.round(lin.intercept_,2)
m = np.round(lin.coef_[0],2)
r2 = np.round(lin.score(datos[['age']], datos['Switch-reference-markers-tail-head']),2)

fig, ax = plt.subplots(dpi=800)
ax.plot(datos[['age']], datos['Switch-reference-markers-tail-head'],marker='o',color='orange',linewidth=0,markersize=5,markeredgewidth=0.25,markeredgecolor='k',alpha=0.85,fillstyle='full',clip_on=True)
ax.plot(datos['age'], lin.predict(datos[['age']]),color='k',linewidth=1.75,markersize=0,markeredgewidth=0.5,markeredgecolor='k',alpha=0.75,fillstyle='full',label=r'$R^2 = $'+str(r2))
plt.grid(False)
plt.legend(loc='best',fontsize=7)
plt.ylabel(r'number of switch reference markers tail head',fontsize=9)
plt.xlabel(r'age',fontsize=12)
plt.rcParams.update({'font.size': 10})
plt.savefig('SRthvsage.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=1024)
plt.show()

## word count versus age
lin = LinearRegression()
lin.fit(datos[['age']], datos['word-count'])
b = np.round(lin.intercept_,2)
m = np.round(lin.coef_[0],2)
r2 = np.round(lin.score(datos[['age']], datos['word-count']),2)

fig, ax = plt.subplots(dpi=800)
ax.plot(datos[['age']], datos['word-count'],marker='o',color='orange',linewidth=0,markersize=5,markeredgewidth=0.25,markeredgecolor='k',alpha=0.85,fillstyle='full',clip_on=True)
ax.plot(datos['age'], lin.predict(datos[['age']]),color='k',linewidth=1.75,markersize=0,markeredgewidth=0.5,markeredgecolor='k',alpha=0.75,fillstyle='full',label=r'$R^2 = $'+str(r2))
plt.grid(False)
plt.legend(loc='best',fontsize=7)
plt.ylabel(r'word count',fontsize=12)
plt.xlabel(r'age',fontsize=12)
plt.rcParams.update({'font.size': 10})
plt.savefig('WCvsage.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=1024)
plt.show()

## nominalizations versus age
lin = LinearRegression()
lin.fit(datos[['age']], datos['nominalizations'])
b = np.round(lin.intercept_,2)
m = np.round(lin.coef_[0],2)
r2 = np.round(lin.score(datos[['age']], datos['nominalizations']),2)

fig, ax = plt.subplots(dpi=800)
ax.plot(datos[['age']], datos['nominalizations'],marker='o',color='orange',linewidth=0,markersize=5,markeredgewidth=0.25,markeredgecolor='k',alpha=0.85,fillstyle='full',clip_on=True)
ax.plot(datos['age'], lin.predict(datos[['age']]),color='k',linewidth=1.75,markersize=0,markeredgewidth=0.5,markeredgecolor='k',alpha=0.75,fillstyle='full',label=r'$R^2 = $'+str(r2))
plt.grid(False)
plt.legend(loc='best',fontsize=7)
plt.ylabel(r'nominalizations',fontsize=12)
plt.xlabel(r'age',fontsize=12)
plt.rcParams.update({'font.size': 10})
plt.savefig('nominalizationsvsage.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=1024)
plt.show()

## sentences versus age
lin = LinearRegression()
lin.fit(datos[['age']], datos['Sentences-count'])
b = np.round(lin.intercept_,2)
m = np.round(lin.coef_[0],2)
r2 = np.round(lin.score(datos[['age']], datos['Sentences-count']),2)

fig, ax = plt.subplots(dpi=800)
ax.plot(datos[['age']], datos['Sentences-count'],marker='o',color='orange',linewidth=0,markersize=5,markeredgewidth=0.25,markeredgecolor='k',alpha=0.85,fillstyle='full',clip_on=True)
ax.plot(datos['age'], lin.predict(datos[['age']]),color='k',linewidth=1.75,markersize=0,markeredgewidth=0.5,markeredgecolor='k',alpha=0.75,fillstyle='full',label=r'$R^2 = $'+str(r2))
plt.grid(False)
plt.legend(loc='best',fontsize=7)
plt.ylabel(r'number of sentences',fontsize=12)
plt.xlabel(r'age',fontsize=12)
plt.rcParams.update({'font.size': 10})
plt.savefig('sentencesvsage.jpg', format='jpg', transparent=True, bbox_inches='tight',dpi=1024)
plt.show()