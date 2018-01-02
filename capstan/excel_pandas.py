import pandas as pd
# from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

#bring in data
df = pd.read_csv("/Users/kasimmonds/Desktop/bird_underway_obs.csv")

#gives species a number
Xuniques, X = np.unique(df['Species'], return_inverse=True)



#marker size dependent on number of birds
min_marker_size=5
def marker_size(numbirds):
    if numbirds < 4:
        return numbirds * min_marker_size
    else:
        return 4 * numbirds

df['msize'] = df['Number birds'].apply(marker_size)
# if df['Number birds'] == 1:
#     msize = min_marker_size
# if df['Number birds'] == 2:
#     msize = 2*min_marker_size
# if df['Number birds'] == 3:
#     msize = 3*min_marker_size
# if df['Number birds'] > 3:
#     msize = 4*min_marker_size


# plt.scatter(df['Latitude'], df['Longitude'], s=df['msize'], c=X)
#location of stations
st_lat = [-34.899, -34.816, -34.703, -34.600, -34.698, -34.610, -34.501, -34.578, -34.573, -34.548, -34.592, -35.224, -35.047]
st_lon = [119.699, 119.499, 119.396, 119.298, 119.697, 119.620, 119.520, 1120.223, 120.205, 120.209, 120.001, 121.679, 121.434]
#
# plt.grid()
# plt.show()

#BASEMAP SCRIPT
plt.figure(1)



#plot map
m = Basemap(projection='cyl', llcrnrlat=-39,urcrnrlat=-31,\
            llcrnrlon=114,urcrnrlon=126, resolution='h',)
x, y = m(df['Longitude'], df['Latitude'])
m.drawparallels(np.array([-39, -37, -35, -33, -31]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([114, 116, 118, 120, 122, 124, 126]), labels=[0,0,0,1], fontsize=7)
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.scatter(x, y, s=df['msize'], c=X)





# labels = [left,right,top,bottom]

xi, yi = m(st_lat, st_lon)
m.plot(yi, xi, marker='D', color='k', markerfacecolor='None', markersize=4, linestyle='None')
#stations = m.plot(st_lon, st_lon, marker='D', color='k', linestyle='None')
plt.savefig('/Users/kasimmonds/Desktop/seabird_loc.png', bbox_inches='tight')
plt.show()
