
#######################################################
####SCRIPT PLOTTING SEABIRDS AGAINST PHYSICAL ENVIRO###
#######################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/kasimmonds/Desktop/bird_underway_obs.csv")

#gives species a number


#change marker size dependent on number of birds
min_marker_size=5
def marker_size(numbirds):
    if numbirds < 4:
        return numbirds * min_marker_size
    else:
        return 4 * numbirds

df['msize'] = df['Number birds'].apply(marker_size)

#define arrays
sal = df['Salinity (PSU)']
print (np.mean(sal))
nbirds = df['Total birds']
do = df['DO']
print (np.mean(do))
flu = df['fluorescence']
print (np.mean(flu))
sst = df['Water Temp']
print (np.mean(sst))
msize = df['msize']
depth = df['Depth']

Xuniques, X = np.unique(df['Bird'], return_inverse=True)

#plot figure
#water temp
plt.figure(1, figsize=(5,10))
ax = plt.subplot(411)
plt.scatter(X, sst, s=msize, c=depth, cmap='winter', linestyle='None')
plt.ylabel('Water Temp.', fontsize=8)
ax.axes.xaxis.set_ticklabels([])
plt.yticks(fontsize=8)
plt.yticks(np.arange(12.5, 22.5, 2.5))

#salinity
ax = plt.subplot(412)
plt.scatter(X, sal, s=msize, c=depth, cmap='winter', linestyle='None')
plt.ylabel('Salinity', fontsize=8)
ax.axes.xaxis.set_ticklabels([])
plt.yticks(fontsize=8)
plt.yticks(np.arange(34.75, 36, 0.25))

#fluoro
ax = plt.subplot(413)
plt.scatter(X, flu, s=msize, c=depth, cmap='winter', linestyle='None')
plt.ylabel('Fluorescence', fontsize=8)
ax.axes.xaxis.set_ticklabels([])
plt.yticks(fontsize=8)
plt.yticks(np.arange(1.4, 3.2, 0.4))

#DO
ax = plt.subplot(414)
plt.scatter(X, do, s=msize, c=depth, cmap='winter', linestyle='None')
plt.ylabel('Dissolved Oxygen', fontsize=8)
ax.axes.xaxis.set_ticklabels([])
plt.yticks(fontsize=8)
plt.yticks(np.arange(245, 310, 10))

#set species name
ax.set(xticks=range(len(Xuniques)), xticklabels=Xuniques)
plt.gcf().subplots_adjust(bottom=0.5)
plt.xticks(fontsize=8, rotation=90)
plt.savefig('/Users/kasimmonds/Desktop/seabird_enviro.png', bbox_inches='tight')
plt.show()
