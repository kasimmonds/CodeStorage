# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
import matplotlib.colors as colors

#ERA/AWAP COMPARISON FOR TNN SON K=9
#era dataset
nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_3_sil_0.1.nc', mode='r')
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

#print 'fOR K=3'
#print 'cluster 0 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c1]), mlat[0], mlon[0])
#print 'cluster 1 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c2]), mlat[1], mlon[1])
#print 'cluster 2 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c3]), mlat[2], mlon[2])



#print stop

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




#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( 0, 0.3, 31, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')



#plot figure
plt.figure(1, figsize=(10,5))
ax = plt.subplot(121)
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

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[0],lon_c1[i]],[mlat[0],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c2)):
  m.plot([mlon[1],lon_c2[i]],[mlat[1],lat_c2[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c3)):
  m.plot([mlon[2],lon_c3[i]],[mlat[2],lat_c3[i]], color='0.8', linewidth=0.75, zorder=1)


mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlon, mmlat, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('1.', xy = (mlon[0],mlat[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('2.', xy = (mlon[1],mlat[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('3.', xy = (mlon[2],mlat[2]), xytext=(5, 5), textcoords='offset points')

cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('a) DJF TNn, K=3', fontsize=12)
cb.ax.tick_params(labelsize=8)
#cb.set_label('Cluster mean silhouette coefficent', fontsize=10)


#k=13 data
nca = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_13_sil_0.1.nc', mode='r')
sila = nca.variables['sil_width'][:,:]
clusta = nca.variables['cluster'][:,:]
mlona = nca.variables['medoid_lon'][:]
mlata = nca.variables['medoid_lat'][:]


#awap k=9 data
#seperate each cluster to determine mean silhouette coefficient
c1a = np.where(clusta == 1)
c2a = np.where(clusta == 2)
c3a = np.where(clusta == 3)
c4a = np.where(clusta == 4)
c5a = np.where(clusta == 5)
c6a = np.where(clusta == 6)
c7a = np.where(clusta == 7)
c8a = np.where(clusta == 8)
c9a = np.where(clusta == 9)
c10a = np.where(clusta == 10)
c11a = np.where(clusta == 11)
c12a = np.where(clusta == 12)
c13a = np.where(clusta == 13)

#for each cluster assign mean sil co value 
s_clusta = np.zeros((clusta.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clusta[i,j] == 1:
      s_clusta[i,j] = np.mean(sila[c1a])
    if clusta[i,j] == 2:
      s_clusta[i,j] = np.mean(sila[c2a])
    if clusta[i,j] == 3:
      s_clusta[i,j] = np.mean(sila[c3a])
    if clusta[i,j] == 4:
      s_clusta[i,j] = np.mean(sila[c4a])
    if clusta[i,j] == 5:
      s_clusta[i,j] = np.mean(sila[c5a])
    if clusta[i,j] == 6:
      s_clusta[i,j] = np.mean(sila[c6a])
    if clusta[i,j] == 7:
      s_clusta[i,j] = np.mean(sila[c7a])
    if clusta[i,j] == 8:
      s_clusta[i,j] = np.mean(sila[c8a])
    if clusta[i,j] == 9:
      s_clusta[i,j] = np.mean(sila[c9a])
    if clusta[i,j] == 10:
      s_clusta[i,j] = np.mean(sila[c10a])
    if clusta[i,j] == 11:
      s_clusta[i,j] = np.mean(sila[c11a])
    if clusta[i,j] == 12:
      s_clusta[i,j] = np.mean(sila[c12a])
    if clusta[i,j] == 13:
      s_clusta[i,j] = np.mean(sila[c13a])
    if s_clusta[i,j] == 0:
      s_clusta[i,j] = np.NaN


#print 'fOR K=13'
##print 'cluster 0 mean = %s' % (np.mean(sila[c1a]))
#print 'cluster 1 mean = %s' % (np.mean(sila[c2a]))
#print 'cluster 2 mean = %s' % (np.mean(sila[c3a]))
##print 'cluster 3 mean = %s' % (np.mean(sila[c4a]))
##print 'cluster 4 mean = %s' % (np.mean(sila[c5a]))
#print 'cluster 5 mean = %s' % (np.mean(sila[c6a]))
##print 'cluster 6 mean = %s' % (np.mean(sila[c7a]))
#print 'cluster 7 mean = %s' % (np.mean(sila[c8a]))
##print 'cluster 8 mean = %s' % (np.mean(sila[c9a]))
#print 'cluster 9 mean = %s' % (np.mean(sila[c10a]))
#print 'cluster 10 mean = %s' % (np.mean(sila[c11a]))
#print 'cluster 11 mean = %s' % (np.mean(sila[c12a]))
#print 'cluster 12 mean = %s' % (np.mean(sila[c13a]))

#make groups for clusters lat/lons
#lons
lon_c1a = []
lon_c2a = []
lon_c3a = []
lon_c4a = []
lon_c5a = []
lon_c6a = []
lon_c7a = []
lon_c8a = []
lon_c9a = []
lon_c10a = []
lon_c11a = []
lon_c12a = []
lon_c13a = []

#lats
lat_c1a = []
lat_c2a = []
lat_c3a = []
lat_c4a = []
lat_c5a = []
lat_c6a = []
lat_c7a = []
lat_c8a = []
lat_c9a = []
lat_c10a = []
lat_c11a = []
lat_c12a = []
lat_c13a = []

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
    if clusta[i,j] == 4 and sila[i,j] > 0.1:
      lon_c4a.append(lon[j])
      lat_c4a.append(lat[i])
    if clusta[i,j] == 5 and sila[i,j] > 0.1:
      lon_c5a.append(lon[j])
      lat_c5a.append(lat[i])
    if clusta[i,j] == 6 and sila[i,j] > 0.1:
      lon_c6a.append(lon[j])
      lat_c6a.append(lat[i])
    if clusta[i,j] == 7 and sila[i,j] > 0.1:
      lon_c7a.append(lon[j])
      lat_c7a.append(lat[i])
    if clusta[i,j] == 8 and sila[i,j] > 0.1:
      lon_c8a.append(lon[j])
      lat_c8a.append(lat[i])
    if clusta[i,j] == 9 and sila[i,j] > 0.1:
      lon_c9a.append(lon[j])
      lat_c9a.append(lat[i])
    if clusta[i,j] == 10 and sila[i,j] > 0.1:
      lon_c10a.append(lon[j])
      lat_c10a.append(lat[i])
    if clusta[i,j] == 11 and sila[i,j] > 0.1:
      lon_c11a.append(lon[j])
      lat_c11a.append(lat[i])
    if clusta[i,j] == 12 and sila[i,j] > 0.1:
      lon_c12a.append(lon[j])
      lat_c12a.append(lat[i])     
    if clusta[i,j] == 13 and sila[i,j] > 0.1:
      lon_c13a.append(lon[j])
      lat_c13a.append(lat[i])       

#plot figure
ax = plt.subplot(122)
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

for i in range(0,len(lon_c4a)):
  m.plot([mlona[3],lon_c4a[i]],[mlata[3],lat_c4a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c5a)):
  m.plot([mlona[4],lon_c5a[i]],[mlata[4],lat_c5a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c6a)):
  m.plot([mlona[5],lon_c6a[i]],[mlata[5],lat_c6a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c7a)):
  m.plot([mlona[6],lon_c7a[i]],[mlata[6],lat_c7a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c8a)):
  m.plot([mlona[7],lon_c8a[i]],[mlata[7],lat_c8a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c9a)):
  m.plot([mlona[8],lon_c9a[i]],[mlata[8],lat_c9a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c10a)):
  m.plot([mlona[9],lon_c10a[i]],[mlata[9],lat_c10a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c11a)):
  m.plot([mlona[10],lon_c11a[i]],[mlata[10],lat_c11a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c12a)):
  m.plot([mlona[11],lon_c12a[i]],[mlata[11],lat_c12a[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c13a)):
  m.plot([mlona[12],lon_c13a[i]],[mlata[12],lat_c13a[i]], color='0.8', linewidth=0.75, zorder=1) 
  
#annotate medoids according to strength
ax.annotate('3.', xy = (mlona[0],mlata[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('8.', xy = (mlona[1],mlata[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('7.', xy = (mlona[2],mlata[2]), xytext=(5, 5), textcoords='offset points')  
ax.annotate('5.', xy = (mlona[3],mlata[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('4.', xy = (mlona[4],mlata[4]), xytext=(5, 5), textcoords='offset points')
ax.annotate('12.', xy = (mlona[5],mlata[5]), xytext=(5, 5), textcoords='offset points')  
ax.annotate('2.', xy = (mlona[6],mlata[6]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlona[7],mlata[7]), xytext=(5, 5), textcoords='offset points')
ax.annotate('1.', xy = (mlona[8],mlata[8]), xytext=(5, 5), textcoords='offset points') 
ax.annotate('13.', xy = (mlona[9],mlata[9]), xytext=(5, 5), textcoords='offset points')
ax.annotate('9.', xy = (mlona[10],mlata[10]), xytext=(5, 5), textcoords='offset points')
ax.annotate('11.', xy = (mlona[11],mlata[11]), xytext=(5, 5), textcoords='offset points') 
ax.annotate('10.', xy = (mlona[12],mlata[12]), xytext=(5, 5), textcoords='offset points') 





mymap= m.scatter(xi, yi, s=msizea, c=s_clusta, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlona, mmlata, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('b) DJF TNn, K=13', fontsize=12)
cb.ax.tick_params(labelsize=8)
cb.set_label('Cluster mean silhouette coefficent', fontsize=10)
plt.savefig('/home/z5147939/hdrive/figs/tnn_djf_3_13_comp.png', bbox_inches='tight')
plt.show()



