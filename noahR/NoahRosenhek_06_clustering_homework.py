'''
CLUSTER ANALYSIS ON COUNTRIES
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the data
dataSet = pd.read_csv('../data/UNdata.csv')
dataUN = pd.DataFrame(dataSet, columns = dataSet.columns)

np.random.seed(0)

# Run KMeans with k = 3
'''
est = KMeans(n_clusters=3)
est.fit(dataSet)
y_kmeans = est.predict(dataSet)
'''

# Use the the following variables: GPDperCapita, lifeMale, lifeFemale, & infantMortality
dataUN2 = dataUN[['country', 'lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]
dataUN2.set_index('country', inplace=True)

'''
# Check if the new index works:
dataUN2.index.is_unique
dataUN2.head()
'''

# Create at least one data visualization (e.g., Scatter plot grid, 3d plot, parallel coordinates)
est = KMeans(n_clusters=3)
est.fit(dataUN2)
y_kmeans = est.predict(dataUN2)

# Sample scatter plot:
colors = np.array(['#FF0054','#FBD039','#23C2BC'])
plt.figure()
plt.scatter(dataUN2['infantMortality'], dataUN2['GDPperCapita'], c=colors[y_kmeans], s=30)
plt.xlabel('infantMortality')
plt.ylabel('GDPperCapita')

# Scatter plot grid:
plt.figure(figsize=(8, 8))
plt.suptitle('Scatter Plot Grid',  fontsize=14)
# Upper Left
plt.subplot(221) # Row, Column, Number
plt.scatter(dataUN2['lifeMale'], dataUN2['GDPperCapita'], c = colors[y_kmeans])
plt.ylabel('GDPperCapita')

# Upper Right
plt.subplot(222)
plt.scatter(dataUN2['lifeFemale'], dataUN2['GDPperCapita'], c = colors[y_kmeans])

# Lower Left
plt.subplot(223)
plt.scatter(dataUN2['lifeMale'], dataUN2['infantMortality'], c = colors[y_kmeans])
plt.ylabel('infantMortality')
plt.xlabel('lifeMale')

# Lower Right
plt.subplot(224)
plt.scatter(dataUN2['lifeFemale'], dataUN2['infantMortality'], c = colors[y_kmeans])
plt.xlabel('lifeFemale')

# Print out the countries present within each cluster. Do you notice any general trend?
dataUN2['cluster'] = y_kmeans
temp = dataUN2.groupby('cluster').groups
print '\nGroup 0: ', temp[0], '\n', '\nGroup 1: ', temp[1], '\n', '\nGroup 2: ', temp[2], '\n'
# I notice that group 0 is made up of countries most people would think of
# as being in the 3rd world, and group 1 is definitely mostly 
# first-world countries. I can't make out what group 2 is..


# Print out the properties of each cluster. What are the most striking differences?
print '\n For each cluster:\n'

temp = dataUN2.groupby('cluster').GDPperCapita.mean()
print '\n The average GDP per Capita is:', temp, '\n'
temp = dataUN2.groupby('cluster').GDPperCapita.std()
print '\n With standard devaiation:', temp, '\n'
temp = dataUN2.groupby('cluster').GDPperCapita.var()
print '\n And variance:', temp, '\n'

temp = dataUN2.groupby('cluster').infantMortality.mean()
print '\n The average rate of infant mortality is:', temp, '\n'
temp = dataUN2.groupby('cluster').infantMortality.std()
print '\n With standard devaiation:', temp, '\n'
temp = dataUN2.groupby('cluster').infantMortality.var()
print '\n And variance:', temp, '\n'


temp = dataUN2.groupby('cluster').lifeMale.mean()
print '\n The average life expectancy for men is: \n', temp, '\n'
temp = dataUN2.groupby('cluster').lifeFemale.mean()
print '\n The average life expectancy for women is: \n', temp, '\n'
temp = dataUN2.groupby('cluster').apply(lambda x: x.lifeMale + x.lifeFemale).mean()
print '\nThe overall average life expectancy is: \n', temp, '\n'

# Advanced: Re-run the cluster analysis after centering and scaling all four variables 
             
# Advanced: How do the results change after they are centered and scaled? Why is this?
             



