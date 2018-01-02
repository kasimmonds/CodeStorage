# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats

had = Dataset('/home/z5147939/ncfiles/sstanom/sst_anom.nc', mode='r')

lon = had.variables['longitude'][:]
lons = had.variables['longitude'][:]
lat = had.variables['latitude'][:]
sst = had.variables['sst'][:,:,:]
time = had.variables['time'][:]
time_units = had.variables['time'].units

#txx

txxf = Dataset('/home/z5147939/ncfiles/sstanom/tmm_fld.nc', mode='r')
tmm = txxf.variables['tmm'][:]


#write the loops + limit for statistical significance  
#define array of zeros

corrtxx = np.zeros((len(lat), len(lon)), dtype=object)
pvtxx = np.zeros((len(lat), len(lon)), dtype=object)

for i in range(0,len(lon)):
    for j in range(0,len(lat)):
            if np.isfinite(np.squeeze(sst[:,j,i])).all():
                corrtxx[j,i], pvtxx[j,i] =  stats.spearmanr((np.squeeze(sst[:,j,i])), np.squeeze(tmm), axis=0)
            else: 
                corrtxx[j,i] = np.NaN

#for correlations < then 0.05, define as stat sig, otherwise = NaN                
ss_txx = np.zeros((len(lat), len(lon)), dtype=object)
for i in range(0,len(lon)):
  for j in range(0,len(lat)):
    if pvtxx[j,i] < 0.05: #if stat sig
        ss_txx[j,i] = corrtxx[j,i]
    else:
        ss_txx[j,i] = np.NaN

#plot
#txx

plt.figure(1)
txx_map = Basemap(projection='cyl', llcrnrlat=-90.0, llcrnrlon=0.0, urcrnrlat=90.0, urcrnrlon=360.0) #define basemap as around Australia
txx_map.drawcoastlines()
txx_map.drawparallels(np.array([-90, -45, 0, 45, 90]), labels=[1,0,0,0], fontsize=10)
txx_map.drawmeridians(np.array([0, 90, 180, 270, 360]), labels=[0,0,0,1], fontsize=10)
corrtxx, lon = shiftgrid(0., corrtxx, lon, start=True)
ss_txx, lon = shiftgrid(0., ss_txx, lons, start=True)
lons, lats = np.meshgrid(lon,lat)
xi,yi = txx_map(lons,lats) 
#v = np.linspace( -0.6, 0.6, 13, endpoint=True) #define colourbar ticks from -0.6 to 0.6
mymap= txx_map.contourf(xi, yi, corrtxx, cmap='bwr')
#ss = txx_map.contourf(xi, yi, ss_txx, v, hatches=['.'], cmap='bwr') #plot ss ontop of correlations
cb = txx_map.colorbar(mymap,"right", size="5%", pad="2%")
cb.ax.tick_params(labelsize=8)
plt.title('TMm', fontsize=16)
#cb.set_label('Correlation')
plt.show()
