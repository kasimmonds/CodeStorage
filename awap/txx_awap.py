# -*- coding: utf-8 -*-


#import neccessary modules
import os
from netCDF4 import Dataset
import numpy as np
#import matplotlib
#matplotlib.use('Agg', warn=False, force=True)
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats#bring in all the variables
import matplotlib.colors as colors
#TMm
#DJF
tdjf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/txx_3mDJF.nc', mode='r')

lon = tdjf.variables['lon'][:]
lat = tdjf.variables['lat'][:]
djf = tdjf.variables['txx'][:,:,:]
time = tdjf.variables['time'][:]

ndjf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/nino_DJF.nc', mode='r')
nino_djf = ndjf.variables['sst'][:]


#write the loops + limit for statistical significance
#define array of zeros

corrdjf = np.zeros((len(lat), len(lon)))
pvdjf = np.zeros((len(lat), len(lon)))
ss_djf = np.zeros((len(lat), len(lon)))
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      corrdjf[j,i], pvdjf[j,i] =  stats.spearmanr((np.squeeze(djf[:,j,i])), np.squeeze(nino_djf), axis=0)
      if pvdjf[j,i] < 0.05: #if stat sig
        ss_djf[j,i] = corrdjf[j,i]
      else:
        ss_djf[j,i] = np.NaN


#MAM
tmam = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/txx_3mMAM.nc', mode='r')
mam = tmam.variables['txx'][:,:,:]


nmam = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/nino_MAM.nc', mode='r')
nino_mam = nmam.variables['sst'][:]


#write the loops + limit for statistical significance

corrmam = np.zeros((len(lat), len(lon)))
pvmam = np.zeros((len(lat), len(lon)))
ss_mam = np.zeros((len(lat), len(lon)))
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      corrmam[j,i], pvmam[j,i] =  stats.spearmanr((np.squeeze(mam[:,j,i])), np.squeeze(nino_mam), axis=0)
      if pvmam[j,i] < 0.05:
        ss_mam[j,i] = corrmam[j,i]
      else:
        ss_mam[j,i] = np.NaN


#JJA
tjja = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/txx_3mJJA.nc', mode='r')
jja = tjja.variables['txx'][:,:,:]


njja = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/nino_JJA.nc', mode='r')
nino_jja = njja.variables['sst'][:]


#write the loops + limit for statistical significance

corrjja = np.zeros((len(lat), len(lon)))
pvjja = np.zeros((len(lat), len(lon)))
ss_jja = np.zeros((len(lat), len(lon)))
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      corrjja[j,i], pvjja[j,i] =  stats.spearmanr((np.squeeze(jja[:,j,i])), np.squeeze(nino_jja), axis=0)
      if pvjja[j,i] < 0.05:
        ss_jja[j,i] = corrjja[j,i]
      else:
        ss_jja[j,i] = np.NaN


#SON
tson = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/txx_3mSON.nc', mode='r')
son = tson.variables['txx'][:,:,:]


nson = Dataset('/srv/ccrc/data06/z5147939/ncfiles/seasonal/nino_SON.nc', mode='r')
nino_son = nson.variables['sst'][:]


#write the loops + limit for statistical significance

corrson = np.zeros((len(lat), len(lon)))
pvson = np.zeros((len(lat), len(lon)))
ss_son = np.zeros((len(lat), len(lon)))
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      corrson[j,i], pvson[j,i] =  stats.spearmanr((np.squeeze(son[:,j,i])), np.squeeze(nino_son), axis=0)
      if pvson[j,i] < 0.05:
        ss_son[j,i] = corrson[j,i]
      else:
        ss_son[j,i] = np.NaN



corrdjf = np.ma.masked_invalid(corrdjf)
corrmam = np.ma.masked_invalid(corrmam)
corrjja = np.ma.masked_invalid(corrjja)
corrson = np.ma.masked_invalid(corrson)


print np.ma.max(corrdjf), np.ma.min(corrdjf)
print np.ma.max(corrmam), np.ma.min(corrmam)
print np.ma.max(corrjja), np.ma.min(corrjja)
print np.ma.max(corrson), np.ma.min(corrson)
#print stop


#general attributes
v = np.linspace( -0.6, 0.6, 13, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)


#plot DJF
fig = plt.figure(1, figsize=(8,3))
ax = plt.subplot(122)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
lons, lats = np.meshgrid(lon,lat)
xi,yi = m(lons,lats)
corrdjf = np.ma.masked_invalid(corrdjf)
ss_djf = np.ma.masked_invalid(ss_djf)
cmap = plt.get_cmap('bwr')
mymap= m.pcolor(xi, yi, corrdjf, norm=norm, cmap=cmap)
ss_djf = np.ma.masked_invalid(ss_djf)
mymap= m.pcolor(xi, yi, ss_djf, hatch = '...', norm=norm, cmap=cmap)
# cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
plt.title('DJF', fontsize=12)

#cb.set_label('Correlation', fontsize=8)


##plot MAM
#plt.subplot(222)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
#mymap= m.contourf(xi, yi, corrmam, v, cmap='bwr')
#ss = m.contourf(xi, yi, ss_mam, v, hatches=['...'], cmap='bwr')
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    #label.set_visible(False)
#plt.title('MAM', fontsize=12)
#cb.ax.tick_params(labelsize=7)
#cb.set_label('Correlation', fontsize=8)

##plot JJA
#plt.subplot(223)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
#mymap= m.contourf(xi, yi, corrjja, v, cmap='bwr')
#ss = m.contourf(xi, yi, ss_jja, v, hatches=['...'], cmap='bwr')
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    #label.set_visible(False)
#plt.title('JJA', fontsize=12)
#cb.ax.tick_params(labelsize=7)
##cb.set_label('Correlation', fontsize=8)

#plot SON
ax = plt.subplot(121)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
corrson = np.ma.masked_invalid(corrson)
mymap = m.pcolor(xi, yi, corrson, norm=norm, cmap=plt.cm.bwr)
ss_son = np.ma.masked_invalid(ss_son)
mymap= m.pcolor(xi, yi, ss_son, hatch = '...', norm=norm, cmap=plt.cm.bwr)
# cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
plt.title('SON', fontsize=12)
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cb = fig.colorbar(mymap, cax, orientation='vertical')
cb.ax.tick_params(labelsize=7)
cb.set_label('Correlation', fontsize=8)
plt.savefig('/home/z5147939/hdrive/figs/txx_awap.png', bbox_inches='tight')
plt.show()
