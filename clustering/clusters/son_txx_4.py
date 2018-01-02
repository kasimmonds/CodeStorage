# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
import matplotlib.colors as colors

#ERA/AWAP COMPARISON FOR TXx SON K=4
#era dataset
nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_hr/txx_SON_K_3_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c1 = np.where(clust == 1)
c2 = np.where(clust == 2)
c3 = np.where(clust == 3)
c4 = np.where(clust == 4)


#for each cluster assign mean sil co value
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 1:
      s_clust[i,j] = np.mean(sil[c1])
    if clust[i,j] == 2:
      s_clust[i,j] = np.mean(sil[c2])
    if clust[i,j] == 3:
      s_clust[i,j] = np.mean(sil[c3])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN

#print 'The ERA max is %s and min is %s' % (np.nanmax(s_clust), np.nanmin(s_clust))
print 'cluster 1 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c1]), mlat[0], mlon[0])
print 'cluster 2 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c2]), mlat[1], mlon[1])
print 'cluster 3 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c3]), mlat[2], mlon[2])

#make groups for clusters lat/lons
#lons
lon_c1 = []
lon_c2 = []
lon_c3 = []

#lats
lat_c1 = []
lat_c2 = []
lat_c3 = []

for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clust[i,j] == 1 and sil[i,j] > 0.1:
      lon_c1.append(lon[j])
      lat_c1.append(lat[i])
    if clust[i,j] == 2 and sil[i,j] > 0.1:
      lon_c2.append(lon[j])
      lat_c2.append(lat[i])
    if clust[i,j] == 3 and sil[i,j] > 0.1:
      lon_c3.append(lon[j])
      lat_c3.append(lat[i])


#awap txx k=4 data
nca = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_hr/txx_ap_SON_K_3_sil_0.1.nc', mode='r')
sila = nca.variables['sil_width'][:,:]
clusta = nca.variables['cluster'][:,:]
mlona = nca.variables['medoid_lon'][:]
mlata = nca.variables['medoid_lat'][:]

#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( 0, 0.28, 29, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)

#plot figure(s)
plt.figure(figsize=(17,7))

ax = plt.subplot(131)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi, yi = m(lons,lats) 
mmlon, mmlat = m(mlon,mlat)
jet = plt.cm.get_cmap('jet')

m.plot(mlon[0],mlat[0], marker='D', color='r')
ax.annotate('2.', xy = (mlon[0],mlat[0]), xytext=(5, 5), textcoords='offset points')
m.plot(mlona[0],mlata[0], marker='o', color='r')

m.plot(mlon[1],mlat[1], marker='D', color='b')
ax.annotate('1.', xy = (mlon[1],mlat[1]), xytext=(5, 5), textcoords='offset points')
m.plot(mlona[1],mlata[1], marker='o', color='b')

m.plot(mlon[2],mlat[2], marker='D', color='w', mew=1.1, label='ERA-Interim')
ax.annotate('3.', xy = (mlon[2],mlat[2]), xytext=(5, 5), textcoords='offset points')
m.plot(mlona[2],mlata[2], marker='o', color='w', mew=1.1, label='AWAP')



plt.legend(numpoints=1, prop={'size':10})
plt.title('a) Medoid Positions for ERA-Interim and AWAP', fontsize=12)

plt.subplot(132)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi, yi = m(lons,lats) 
mmlon, mmlat = m(mlon,mlat)
jet = plt.cm.get_cmap('jet')

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
  m.plot([mlon[0],lon_c1[i]],[mlat[0],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c2)):
  m.plot([mlon[1],lon_c2[i]],[mlat[1],lat_c2[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c3)):
  m.plot([mlon[2],lon_c3[i]],[mlat[2],lat_c3[i]], color='0.8', linewidth=0.75, zorder=1)


mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlon, mmlat, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('b) ERA-Interim SON TXx, K=3', fontsize=12)
cb.ax.tick_params(labelsize=8)
#cb.set_label('Cluster mean silhouette coefficent', fontsize=10)


#AWAP

#seperate each cluster to determine mean silhouette coefficient
c1a = np.where(clusta == 1)
c2a = np.where(clusta == 2)
c3a = np.where(clusta == 3)



#for each cluster assign mean sil co value 
s_clusta = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clusta[i,j] == 1:
      s_clusta[i,j] = np.mean(sila[c1a])
    if clusta[i,j] == 2:
      s_clusta[i,j] = np.mean(sila[c2a])
    if clusta[i,j] == 3:
      s_clusta[i,j] = np.mean(sila[c3a])
    if s_clusta[i,j] == 0:
      s_clusta[i,j] = np.NaN


print 'The AWAP max is %s and min is %s' % (np.nanmax(s_clusta), np.nanmin(s_clusta))

#make groups for clusters lat/lons
#lons
lon_c1a = []
lon_c2a = []
lon_c3a = []


#lats
lat_c1a = []
lat_c2a = []
lat_c3a = []


for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clusta[i,j] == 1 and sila[i,j] > 0.1:
      lon_c1a.append(lon[j])
      lat_c1a.append(lat[i])
    if clusta[i,j] == 2 and sila[i,j] > 0.1:
      lon_c2a.append(lon[j])
      lat_c2a.append(lat[i])
    if clusta[i,j] == 3 and sila[i,j] > 0.1:
      lon_c3a.append(lon[j])
      lat_c3a.append(lat[i])



#plot figure
plt.subplot(133)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi, yi = m(lons,lats) 
mmlona, mmlata = m(mlona,mlata)

#large markers for significant values, small markers for insignificant 
min_marker_size = 3
msizea = np.zeros((sila.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sila[i,j] > 0.1:
      msizea[i,j] = 3 * min_marker_size
    else:
      msizea[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1a)):
  m.plot([mlona[0],lon_c1a[i]],[mlata[0],lat_c1a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c2a)):
  m.plot([mlona[1],lon_c2a[i]],[mlata[1],lat_c2a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c3a)):
  m.plot([mlona[2],lon_c3a[i]],[mlata[2],lat_c3a[i]], color='0.8', linewidth=0.75, zorder=1)



mymap= m.scatter(xi, yi, s=msizea, c=s_clusta, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlona, mmlata, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('c) AWAP SON TXx, K=3', fontsize=12)
cb.ax.tick_params(labelsize=8)
cb.set_label('Cluster mean silhouette coefficent', fontsize=10)
plt.savefig('/home/z5147939/hdrive/figs/txx_son_4clust.png', bbox_inches='tight')
plt.show()


