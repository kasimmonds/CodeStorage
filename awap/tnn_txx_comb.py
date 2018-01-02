#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats
import matplotlib.colors as colors



#SON
tson = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/tnn_3m_SON.nc', mode='r')
tnn = tson.variables['tnn'][:,:,:]
lon = tson.variables['lon'][:]
lat = tson.variables['lat'][:]


txson = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/txx_3mSON.nc', mode='r')
txx = txson.variables['txx'][:,:,:]


nson = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/nino_SON.nc', mode='r')
nino_son = nson.variables['sst'][:]


#write the loops + limit for statistical significance

corrtnn = np.zeros((len(lat), len(lon)))
pvtnn = np.zeros((len(lat), len(lon)))
ss_tnn = np.zeros((len(lat), len(lon)))

corrtxx = np.zeros((len(lat), len(lon)))
pvtxx = np.zeros((len(lat), len(lon)))
ss_txx = np.zeros((len(lat), len(lon)))

for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      corrtnn[j,i], pvtnn[j,i] =  stats.spearmanr((np.squeeze(tnn[:,j,i])), np.squeeze(nino_son), axis=0)
      if pvtnn[j,i] < 0.05:
        ss_tnn[j,i] = corrtnn[j,i]
      else:
        ss_tnn[j,i] = np.NaN
      corrtxx[j,i], pvtxx[j,i] =  stats.spearmanr((np.squeeze(txx[:,j,i])), np.squeeze(nino_son), axis=0)
      if pvtxx[j,i] < 0.05:
        ss_txx[j,i] = corrtxx[j,i]
      else:
        ss_txx[j,i] = np.NaN




#print stop


#plot
v = np.linspace( -0.5, 0.5, 13, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
#plot DJF
fig = plt.figure(1, figsize=(8,3))
ax = plt.subplot(111)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()

lons, lats = np.meshgrid(lon,lat)
xi,yi = m(lons,lats)
#mymap = m.contourf(xi, yi, corrdjf, v)
ss_txx = np.ma.masked_invalid(ss_txx)
ss_tnn = np.ma.masked_invalid(ss_tnn)
#mymap= m.pcolor(xi, yi, corrdjf, norm=norm, cmap=plt.cm.bwr)
mymap= m.pcolor(xi, yi, ss_tnn,  norm=norm, cmap=plt.cm.bwr)
mymap= m.pcolor(xi, yi, ss_txx,  norm=norm, cmap=plt.cm.bwr)

plt.savefig('/home/z5147939/hdrive/figs/tnn_txx.png', bbox_inches='tight')
plt.show()
