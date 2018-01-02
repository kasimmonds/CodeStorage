# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid, maskoceans
import matplotlib.colors as colors
from scipy import stats

#set out cluster
nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#bring in tnn dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/tnn_ei_3m_DJF.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]

def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx

lat_idx_n4 = geo_idx(mlat[3], lats)
lon_idx_n4 = geo_idx(mlon[3], lons)

tnn_djf_c6 = ts.variables['tnn'][:,lat_idx_n4,lon_idx_n4]

#ssta
ssta = Dataset('/srv/ccrc/data06/z5147939/ncfiles/sstanom/ssta_corr_DJF.nc', mode='r')
sst_djf = ssta.variables['sst'][:,:,:]
lonx = ssta.variables['longitude'][:]
latx = ssta.variables['latitude'][:]

corr_c1 = np.zeros((len(latx), len(lonx)))
pv_c1 = np.zeros((len(latx), len(lonx)))
ss_c1 = np.zeros((len(latx), len(lonx)))

for i in range(0,len(lonx)):
    for j in range(0,len(latx)):
      #n1
      corr_c1[j,i], pv_c1[j,i] =  stats.spearmanr((np.squeeze(sst_djf[:,j,i])), np.squeeze(tnn_djf_c6), axis=0)
      if pv_c1[j,i] < 0.05: #if stat sig
        ss_c1[j,i] = corr_c1[j,i]
      else:
        ss_c1[j,i] = np.NaN

#pick cluster 1
c1 = np.where(clust == 4)
lon_c1 = []
lat_c1 = []

#for each cluster assign mean sil co value and put lat and lons of that cluster into lists
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 4:
      s_clust[i,j] = np.mean(sil[c1])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN
    if clust[i,j] == 4 and sil[i,j] > 0.1:
      lon_c1.append(lon[j])
      lat_c1.append(lat[i])

#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.8, 0.8, 17, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')
#plot figure
fig = plt.figure(1, figsize=(9,5))
ax = plt.subplot(233)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
xi, yi = m(lons,lats)

#large markers for significant values, small markers for insignificant
min_marker_size = 3
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[3],lon_c1[i]],[mlat[3],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

points = m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap='jet', edgecolors='none', zorder=2)
medoid = m.plot(mlon[3], mlat[3], 'D', color='k', fillstyle='none', mew=2, markersize=3)

plt.title('TNn DJF Cluster 6', fontsize=10)
lons, lats = np.meshgrid(lonx,latx)
xi, yi = m(lons,lats)
corr_c1 = np.ma.masked_invalid(corr_c1)
ss_c1 = np.ma.masked_invalid(ss_c1)
mymap= m.pcolormesh(xi, yi, corr_c1, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_c1, hatch='...', norm=norm, cmap='bwr')

#TXX DJF CLUSTER

nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#bring in tnn dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/txx_ei_3m_DJF.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]

def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx

lat_idx_n4 = geo_idx(mlat[6], lats)
lon_idx_n4 = geo_idx(mlon[6], lons)

txx_djf_c6 = ts.variables['txx'][:,lat_idx_n4,lon_idx_n4]


corr_c2 = np.zeros((len(latx), len(lonx)))
pv_c2 = np.zeros((len(latx), len(lonx)))
ss_c2 = np.zeros((len(latx), len(lonx)))

for i in range(0,len(lonx)):
    for j in range(0,len(latx)):
      #n1
      corr_c2[j,i], pv_c2[j,i] =  stats.spearmanr((np.squeeze(sst_djf[:,j,i])), np.squeeze(txx_djf_c6), axis=0)
      if pv_c2[j,i] < 0.05: #if stat sig
        ss_c2[j,i] = corr_c2[j,i]
      else:
        ss_c2[j,i] = np.NaN

#pick cluster 1
c1 = np.where(clust == 7)
lon_c1 = []
lat_c1 = []

#for each cluster assign mean sil co value and put lat and lons of that cluster into lists
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 7:
      s_clust[i,j] = np.mean(sil[c1])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN
    if clust[i,j] == 7 and sil[i,j] > 0.1:
      lon_c1.append(lon[j])
      lat_c1.append(lat[i])

#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.8, 0.8, 17, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')
#plot figure

ax = plt.subplot(236)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
xi, yi = m(lons,lats)

#large markers for significant values, small markers for insignificant
min_marker_size = 3
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[6],lon_c1[i]],[mlat[6],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

points = m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap='rainbow', edgecolors='none', zorder=2)
medoid = m.plot(mlon[6], mlat[6], 'D', color='k', fillstyle='none', mew=2, markersize=3)

plt.title('TXx DJF Cluster 6', fontsize=10)
lons, lats = np.meshgrid(lonx,latx)
xi, yi = m(lons,lats)
corr_c2 = np.ma.masked_invalid(corr_c2)
ss_c2 = np.ma.masked_invalid(ss_c2)
mymap= m.pcolormesh(xi, yi, corr_c2, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_c2, hatch='...', norm=norm, cmap='bwr')

#Tnn son cluster 2 CLUSTER

nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#bring in tnn dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/tnn_ei_3m_SON.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]

def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx

lat_idx_n4 = geo_idx(mlat[4], lats)
lon_idx_n4 = geo_idx(mlon[4], lons)

tnn_son_c2 = ts.variables['tnn'][:,lat_idx_n4,lon_idx_n4]

#ssta
ssta = Dataset('/srv/ccrc/data06/z5147939/ncfiles/sstanom/ssta_corr_SON.nc', mode='r')
sst_son = ssta.variables['sst'][:,:,:]

corr_c3 = np.zeros((len(latx), len(lonx)))
pv_c3 = np.zeros((len(latx), len(lonx)))
ss_c3 = np.zeros((len(latx), len(lonx)))

for i in range(0,len(lonx)):
    for j in range(0,len(latx)):
      #n1
      corr_c3[j,i], pv_c3[j,i] =  stats.spearmanr((np.squeeze(sst_son[:,j,i])), np.squeeze(tnn_son_c2), axis=0)
      if pv_c3[j,i] < 0.05: #if stat sig
        ss_c3[j,i] = corr_c3[j,i]
      else:
        ss_c3[j,i] = np.NaN

#pick cluster 1
c1 = np.where(clust == 5)
lon_c1 = []
lat_c1 = []

#for each cluster assign mean sil co value and put lat and lons of that cluster into lists
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 5:
      s_clust[i,j] = np.mean(sil[c1])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN
    if clust[i,j] == 5 and sil[i,j] > 0.1:
      lon_c1.append(lon[j])
      lat_c1.append(lat[i])

#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.8, 0.8, 17, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')
#plot figure

ax = plt.subplot(231)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
xi, yi = m(lons,lats)

#large markers for significant values, small markers for insignificant
min_marker_size = 3
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[4],lon_c1[i]],[mlat[4],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

points = m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap='rainbow', edgecolors='none', zorder=2)
medoid = m.plot(mlon[4], mlat[4], 'D', color='k', fillstyle='none', mew=2, markersize=3)

plt.title('TNn SON Cluster 2', fontsize=10)
lons, lats = np.meshgrid(lonx,latx)
xi, yi = m(lons,lats)
corr_c3 = np.ma.masked_invalid(corr_c3)
ss_c3 = np.ma.masked_invalid(ss_c3)
mymap= m.pcolormesh(xi, yi, corr_c3, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_c3, hatch='...', norm=norm, cmap='bwr')

#Txx son cluster 7 CLUSTER

nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#bring in tnn dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/txx_ei_3m_SON.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]

def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx

lat_idx_n4 = geo_idx(mlat[4], lats)
lon_idx_n4 = geo_idx(mlon[4], lons)

txx_son_c7 = ts.variables['txx'][:,lat_idx_n4,lon_idx_n4]


corr_c4 = np.zeros((len(latx), len(lonx)))
pv_c4 = np.zeros((len(latx), len(lonx)))
ss_c4 = np.zeros((len(latx), len(lonx)))

for i in range(0,len(lonx)):
    for j in range(0,len(latx)):
      #n1
      corr_c4[j,i], pv_c4[j,i] =  stats.spearmanr((np.squeeze(sst_son[:,j,i])), np.squeeze(txx_son_c7), axis=0)
      if pv_c4[j,i] < 0.05: #if stat sig
        ss_c4[j,i] = corr_c4[j,i]
      else:
        ss_c4[j,i] = np.NaN

#pick cluster 1
c1 = np.where(clust == 5)
lon_c1 = []
lat_c1 = []

#for each cluster assign mean sil co value and put lat and lons of that cluster into lists
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 5:
      s_clust[i,j] = np.mean(sil[c1])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN
    if clust[i,j] == 5 and sil[i,j] > 0.1:
      lon_c1.append(lon[j])
      lat_c1.append(lat[i])

#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.8, 0.8, 17, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')
#plot figure

ax = plt.subplot(234)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
xi, yi = m(lons,lats)

#large markers for significant values, small markers for insignificant
min_marker_size = 3
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[4],lon_c1[i]],[mlat[4],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

points = m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap='rainbow', edgecolors='none', zorder=2)
medoid = m.plot(mlon[4], mlat[4], 'D', color='k', fillstyle='none', mew=2, markersize=3)

plt.title('TXx SON Cluster 7', fontsize=10)
lons, lats = np.meshgrid(lonx,latx)
xi, yi = m(lons,lats)
corr_c4 = np.ma.masked_invalid(corr_c4)
ss_c4 = np.ma.masked_invalid(ss_c4)
mymap= m.pcolormesh(xi, yi, corr_c4, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_c4, hatch='...', norm=norm, cmap='bwr')

#Tnn son cluster 7 CLUSTER

nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#bring in tnn dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/tnn_ei_3m_SON.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]

def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx

lat_idx_n4 = geo_idx(mlat[6], lats)
lon_idx_n4 = geo_idx(mlon[6], lons)

tnn_son_c7 = ts.variables['tnn'][:,lat_idx_n4,lon_idx_n4]


corr_c5 = np.zeros((len(latx), len(lonx)))
pv_c5 = np.zeros((len(latx), len(lonx)))
ss_c5 = np.zeros((len(latx), len(lonx)))

for i in range(0,len(lonx)):
    for j in range(0,len(latx)):
      #n1
      corr_c5[j,i], pv_c5[j,i] =  stats.spearmanr((np.squeeze(sst_son[:,j,i])), np.squeeze(tnn_son_c7), axis=0)
      if pv_c5[j,i] < 0.05: #if stat sig
        ss_c5[j,i] = corr_c5[j,i]
      else:
        ss_c5[j,i] = np.NaN

#pick cluster 1
c1 = np.where(clust == 7)
lon_c1 = []
lat_c1 = []

#for each cluster assign mean sil co value and put lat and lons of that cluster into lists
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 7:
      s_clust[i,j] = np.mean(sil[c1])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN
    if clust[i,j] == 7 and sil[i,j] > 0.1:
      lon_c1.append(lon[j])
      lat_c1.append(lat[i])

#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.8, 0.8, 17, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')
#plot figure

ax = plt.subplot(232)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
xi, yi = m(lons,lats)

#large markers for significant values, small markers for insignificant
min_marker_size = 3
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[6],lon_c1[i]],[mlat[6],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

points = m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap='rainbow', edgecolors='none', zorder=2)
medoid = m.plot(mlon[6], mlat[6], 'D', color='k', fillstyle='none', mew=2, markersize=3)

plt.title('TNn SON Cluster 7', fontsize=10)
lons, lats = np.meshgrid(lonx,latx)
xi, yi = m(lons,lats)
corr_c5 = np.ma.masked_invalid(corr_c5)
ss_c5 = np.ma.masked_invalid(ss_c5)
mymap= m.pcolormesh(xi, yi, corr_c5, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_c5, hatch='...', norm=norm, cmap='bwr')

#Txx son cluster 2 CLUSTER

nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#bring in tnn dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/txx_ei_3m_SON.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]

def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx

lat_idx_n4 = geo_idx(mlat[5], lats)
lon_idx_n4 = geo_idx(mlon[5], lons)

txx_son_c2 = ts.variables['txx'][:,lat_idx_n4,lon_idx_n4]


corr_c6 = np.zeros((len(latx), len(lonx)))
pv_c6 = np.zeros((len(latx), len(lonx)))
ss_c6 = np.zeros((len(latx), len(lonx)))

for i in range(0,len(lonx)):
    for j in range(0,len(latx)):
      #n1
      corr_c6[j,i], pv_c6[j,i] =  stats.spearmanr((np.squeeze(sst_son[:,j,i])), np.squeeze(txx_son_c2), axis=0)
      if pv_c6[j,i] < 0.05: #if stat sig
        ss_c6[j,i] = corr_c6[j,i]
      else:
        ss_c6[j,i] = np.NaN

#pick cluster 1
c1 = np.where(clust == 6)
lon_c1 = []
lat_c1 = []

#for each cluster assign mean sil co value and put lat and lons of that cluster into lists
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 6:
      s_clust[i,j] = np.mean(sil[c1])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN
    if clust[i,j] == 6 and sil[i,j] > 0.1:
      lon_c1.append(lon[j])
      lat_c1.append(lat[i])

#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.8, 0.8, 17, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')
#plot figure

ax = plt.subplot(235)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
xi, yi = m(lons,lats)

#large markers for significant values, small markers for insignificant
min_marker_size = 3
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[5],lon_c1[i]],[mlat[5],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

points = m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap='rainbow', edgecolors='none', zorder=2)
medoid = m.plot(mlon[5], mlat[5], 'D', color='k', fillstyle='none', mew=2, markersize=3)

plt.title('TXx SON Cluster 2', fontsize=10)
lons, lats = np.meshgrid(lonx,latx)
xi, yi = m(lons,lats)
corr_c6 = np.ma.masked_invalid(corr_c6)
ss_c6 = np.ma.masked_invalid(ss_c6)
mymap= m.pcolormesh(xi, yi, corr_c6, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_c6, hatch='...', norm=norm, cmap='bwr')


cax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
cb = fig.colorbar(mymap, cax, orientation='vertical')
cb.ax.tick_params(labelsize=7)
cb.set_label('Correlation', fontsize=8)
plt.savefig('/home/z5147939/hdrive/figs/ssta_corr.png', bbox_inches='tight')
plt.show()
