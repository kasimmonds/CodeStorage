# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
import matplotlib.colors as colors

############################
###SON effected by enso
############################

#tnn
nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_7_sil_0.1.nc', mode='r')
silx = nce.variables['sil_width'][:,:]
clustx = nce.variables['cluster'][:,:]
mlonx = nce.variables['medoid_lon'][:]
mlatx = nce.variables['medoid_lat'][:]
#seperate each cluster to determine mean silhouette coefficient
sn2 = np.where(clust == 5)
sx2 = np.where(clustx == 6)
sx5 = np.where(clustx == 4)
sx6 = np.where(clustx == 7)

#for each cluster assign mean sil co value
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 5:
      s_clust[i,j] = 1
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN


s_clustx = np.zeros((clustx.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustx[i,j] == 6:
      s_clustx[i,j] = 4
    if clustx[i,j] == 4:
      s_clustx[i,j] = 9
    if clustx[i,j] == 7:
      s_clustx[i,j] = 12
    if s_clustx[i,j] == 0:
      s_clustx[i,j] = np.NaN


#make groups for clusters lat/lons
lon_sn2 = []
lat_sn2 = []

for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 5 and sil[i,j] > 0.1:
      lon_sn2.append(lon[j])
      lat_sn2.append(lat[i])

lon_sx2 = []
lat_sx2 = []
lon_sx5 = []
lat_sx5 = []
lon_sx6 = []
lat_sx6 = []

for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustx[i,j] == 6 and silx[i,j] > 0.1:
      lon_sx2.append(lon[j])
      lat_sx2.append(lat[i])
    if clustx[i,j] == 4 and silx[i,j] > 0.1:
      lon_sx5.append(lon[j])
      lat_sx5.append(lat[i])
    if clustx[i,j] == 7 and silx[i,j] > 0.1:
      lon_sx6.append(lon[j])
      lat_sx6.append(lat[i])


#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -5, 20, 26, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')



#plot figure
plt.figure(1, figsize=(7,5))
ax = plt.subplot(111)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi, yi = m(lons,lats)
mmlon, mmlat = m(mlon,mlat)
jet = plt.cm.get_cmap('jet')

#large markers for significant values, small markers for insignificant
min_marker_size = 5
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

msizex = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if silx[i,j] > 0.1:
      msizex[i,j] = 3 * min_marker_size
    else:
      msizex[i,j] =  min_marker_size

#plot lines between significant values and medoids

for i in range(0,len(lon_sn2)):
  m.plot([mlon[4],lon_sn2[i]],[mlat[4],lat_sn2[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sx2)):
  m.plot([mlonx[5],lon_sx2[i]],[mlatx[5],lat_sx2[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sx5)):
  m.plot([mlonx[3],lon_sx5[i]],[mlatx[3],lat_sx5[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sx6)):
  m.plot([mlonx[6],lon_sx6[i]],[mlatx[6],lat_sx6[i]], color='0.8', linewidth=0.75, zorder=1)

mymap= m.scatter(xi, yi, s=msizex, c=s_clustx, norm=norm, cmap=jet, edgecolors='none', zorder=2)
mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlon[4], mlat[4], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonx[5], mlatx[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonx[3], mlatx[3], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonx[6], mlatx[6], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
# ax.annotate('TXx cluster 2', xy = (mlonx[5],mlatx[5]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# ax.annotate('TNn cluster 2', xy = (mlon[4],mlat[4]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# ax.annotate('TXx cluster 5', xy = (mlonx[3],mlatx[3]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# ax.annotate('TXx cluster 6', xy = (mlonx[6],mlatx[6]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# plt.title('SON', fontsize=20)
plt.savefig('/home/z5147939/hdrive/figs/clust_enso_soni.png', bbox_inches='tight')
plt.show()

#cluster associated for son

sn4 = np.where(clust == 6)
sn6 = np.where(clust == 4)
sn7 = np.where(clust == 7)
sx7 = np.where(clustx == 5)

#for each cluster assign mean sil co value
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 6:
      s_clust[i,j] = 3
    if clust[i,j] == 4:
      s_clust[i,j] = 8
    if clust[i,j] == 7:
      s_clust[i,j] = 14
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN

s_clustx = np.zeros((clustx.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustx[i,j] == 5:
      s_clustx[i,j] = 11
    if s_clustx[i,j] == 0:
      s_clustx[i,j] = np.NaN


#make groups for clusters lat/lons
lon_sn4 = []
lat_sn4 = []
lon_sn6 = []
lat_sn6 = []
lon_sn7 = []
lat_sn7 = []

for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 6 and sil[i,j] > 0.1:
      lon_sn4.append(lon[j])
      lat_sn4.append(lat[i])
    if clust[i,j] == 4 and sil[i,j] > 0.1:
      lon_sn6.append(lon[j])
      lat_sn6.append(lat[i])
    if clust[i,j] == 7 and sil[i,j] > 0.1:
      lon_sn7.append(lon[j])
      lat_sn7.append(lat[i])

lon_sx7 = []
lat_sx7 = []


for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustx[i,j] == 5 and silx[i,j] > 0.1:
      lon_sx7.append(lon[j])
      lat_sx7.append(lat[i])




#plot figure
plt.figure(2, figsize=(7,5))
ax = plt.subplot(111)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
# xi, yi = m(lons,lats)
# mmlon, mmlat = m(mlon,mlat)


#large markers for significant values, small markers for insignificant


#plot lines between significant values and medoids

for i in range(0,len(lon_sn4)):
  m.plot([mlon[5],lon_sn4[i]],[mlat[5],lat_sn4[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sn6)):
  m.plot([mlon[3],lon_sn6[i]],[mlat[3],lat_sn6[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sn7)):
  m.plot([mlon[6],lon_sn7[i]],[mlat[6],lat_sn7[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sx7)):
  m.plot([mlonx[4],lon_sx7[i]],[mlatx[4],lat_sx7[i]], color='0.8', linewidth=0.75, zorder=1)

mymap= m.scatter(xi, yi, s=msizex, c=s_clustx, norm=norm, cmap=jet, edgecolors='none', zorder=2)
mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlon[5], mlat[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlon[3], mlat[3], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlon[6], mlat[6], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonx[4], mlatx[4], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
# ax.annotate('TNn cluster 4', xy = (mlon[5],mlat[5]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# ax.annotate('TNn cluster 6', xy = (mlon[3],mlat[3]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# ax.annotate('TNn cluster 7', xy = (mlon[6],mlat[6]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# ax.annotate('TXx cluster 7', xy = (mlonx[4],mlatx[4]), xytext=(5, 5), textcoords='offset points', fontsize=12)
# plt.title('SON', fontsize=20)
plt.savefig('/home/z5147939/hdrive/figs/clust_notenso_soni.png', bbox_inches='tight')
plt.show()

############################
###djf effected by enso
############################

#tnn
nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_7_sil_0.1.nc', mode='r')
silx = nce.variables['sil_width'][:,:]
clustx = nce.variables['cluster'][:,:]
mlonx = nce.variables['medoid_lon'][:]
mlatx = nce.variables['medoid_lat'][:]
#seperate each cluster to determine mean silhouette coefficient
sx2 = np.where(clust == 1)
sx4 = np.where(clustx == 6)
sx6 = np.where(clustx == 7)


#for each cluster assign mean sil co value


s_clustx = np.zeros((clustx.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustx[i,j] == 1:
      s_clustx[i,j] = 2
    if clustx[i,j] == 6:
      s_clustx[i,j] = 6
    if clustx[i,j] == 7:
      s_clustx[i,j] = 12
    if s_clustx[i,j] == 0:
      s_clustx[i,j] = np.NaN


#make groups for clusters lat/lons

lon_sx2 = []
lat_sx2 = []
lon_sx4 = []
lat_sx4 = []
lon_sx6 = []
lat_sx6 = []

for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustx[i,j] == 1 and silx[i,j] > 0.1:
      lon_sx2.append(lon[j])
      lat_sx2.append(lat[i])
    if clustx[i,j] == 6 and silx[i,j] > 0.1:
      lon_sx4.append(lon[j])
      lat_sx4.append(lat[i])
    if clustx[i,j] == 7 and silx[i,j] > 0.1:
      lon_sx6.append(lon[j])
      lat_sx6.append(lat[i])

#plot figure
plt.figure(1, figsize=(7,5))
ax = plt.subplot(111)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)


#large markers for significant values, small markers for insignificant
min_marker_size = 5
msize = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sil[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

msizex = np.zeros((sil.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if silx[i,j] > 0.1:
      msizex[i,j] = 3 * min_marker_size
    else:
      msizex[i,j] =  min_marker_size

#plot lines between significant values and medoids


for i in range(0,len(lon_sx2)):
  m.plot([mlonx[0],lon_sx2[i]],[mlatx[0],lat_sx2[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sx4)):
  m.plot([mlonx[5],lon_sx4[i]],[mlatx[5],lat_sx4[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sx6)):
  m.plot([mlonx[6],lon_sx6[i]],[mlatx[6],lat_sx6[i]], color='0.8', linewidth=0.75, zorder=1)

mymap= m.scatter(xi, yi, s=msizex, c=s_clustx, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlonx[0], mlatx[0], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonx[5], mlatx[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonx[6], mlatx[6], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('TXx cluster 2', xy = (mlonx[0],mlatx[0]), xytext=(5, 5), textcoords='offset points', fontsize=12)
ax.annotate('TXx cluster 4', xy = (mlonx[5],mlatx[5]), xytext=(5, 5), textcoords='offset points', fontsize=12)
ax.annotate('TXx cluster 6', xy = (mlonx[6],mlatx[6]), xytext=(5, 5), textcoords='offset points', fontsize=12)
plt.title('DJF', fontsize=20)
plt.savefig('/home/z5147939/hdrive/figs/clust_enso_djf.png', bbox_inches='tight')
plt.show()

#cluster associated for djf

sn1 = np.where(clust == 1)
sn5 = np.where(clust == 6)
sn6 = np.where(clust == 4)


#for each cluster assign mean sil co value
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 1:
      s_clust[i,j] = 1
    if clust[i,j] == 6:
      s_clust[i,j] = 10
    if clust[i,j] == 4:
      s_clust[i,j] = 13
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN




#make groups for clusters lat/lons
lon_sn1 = []
lat_sn1 = []
lon_sn5 = []
lat_sn5 = []
lon_sn6 = []
lat_sn6 = []

for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 1 and sil[i,j] > 0.1:
      lon_sn1.append(lon[j])
      lat_sn1.append(lat[i])
    if clust[i,j] == 6 and sil[i,j] > 0.1:
      lon_sn5.append(lon[j])
      lat_sn5.append(lat[i])
    if clust[i,j] == 4 and sil[i,j] > 0.1:
      lon_sn6.append(lon[j])
      lat_sn6.append(lat[i])

#plot figure
plt.figure(4, figsize=(7,5))
ax = plt.subplot(111)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)

#plot lines between significant values and medoids

for i in range(0,len(lon_sn1)):
  m.plot([mlon[0],lon_sn1[i]],[mlat[0],lat_sn1[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sn5)):
  m.plot([mlon[5],lon_sn5[i]],[mlat[5],lat_sn5[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_sn6)):
  m.plot([mlon[3],lon_sn6[i]],[mlat[3],lat_sn6[i]], color='0.8', linewidth=0.75, zorder=1)

mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlon[0], mlat[0], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlon[5], mlat[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlon[3], mlat[3], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)


#annotate medoids according to strength
ax.annotate('TNn cluster 1', xy = (mlon[0],mlat[0]), xytext=(5, 5), textcoords='offset points', fontsize=12)
ax.annotate('TNn cluster 5', xy = (mlon[5],mlat[5]), xytext=(5, 5), textcoords='offset points', fontsize=12)
ax.annotate('TNn cluster 6', xy = (mlon[3],mlat[3]), xytext=(5, 5), textcoords='offset points', fontsize=12)
plt.title('DJF', fontsize=20)
plt.savefig('/home/z5147939/hdrive/figs/clust_notenso_djf.png', bbox_inches='tight')
plt.show()
