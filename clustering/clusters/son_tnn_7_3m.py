# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
import matplotlib.colors as colors

#ERA/AWAP COMPARISON FOR TNN SON K=9
#era dataset
nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_7_sil_0.1.nc', mode='r')
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
c5 = np.where(clust == 5)
c6 = np.where(clust == 6)
c7 = np.where(clust == 7)


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
    if clust[i,j] == 4:
      s_clust[i,j] = np.mean(sil[c4])
    if clust[i,j] == 5:
      s_clust[i,j] = np.mean(sil[c5])
    if clust[i,j] == 6:
      s_clust[i,j] = np.mean(sil[c6])
    if clust[i,j] == 7:
      s_clust[i,j] = np.mean(sil[c7])
    if s_clust[i,j] == 0:
      s_clust[i,j] = np.NaN

##print 'The ERA max is %s and min is %s' % (np.nanmax(s_clust), np.nanmin(s_clust))
print 'cluster 0 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c1]), mlat[0], mlon[0])
print 'cluster 1 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c2]), mlat[1], mlon[1])
print 'cluster 2 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c3]), mlat[2], mlon[2])
print 'cluster 3 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c4]), mlat[3], mlon[3])
print 'cluster 4 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c5]), mlat[4], mlon[4])
print 'cluster 5 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c6]), mlat[5], mlon[5])
print 'cluster 6 mean = %s, lat = %s, lon = %s' % (np.mean(sil[c7]), mlat[6], mlon[6])


#print stop

#make groups for clusters lat/lons
#lons
lon_c1 = []
lon_c2 = []
lon_c3 = []
lon_c4 = []
lon_c5 = []
lon_c6 = []
lon_c7 = []


#lats
lat_c1 = []
lat_c2 = []
lat_c3 = []
lat_c4 = []
lat_c5 = []
lat_c6 = []
lat_c7 = []


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
    if clust[i,j] == 4 and sil[i,j] > 0.1:
      lon_c4.append(lon[j])
      lat_c4.append(lat[i])
    if clust[i,j] == 5 and sil[i,j] > 0.1:
      lon_c5.append(lon[j])
      lat_c5.append(lat[i])
    if clust[i,j] == 6 and sil[i,j] > 0.1:
      lon_c6.append(lon[j])
      lat_c6.append(lat[i])
    if clust[i,j] == 7 and sil[i,j] > 0.1:
      lon_c7.append(lon[j])
      lat_c7.append(lat[i])



#global definitions
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( 0, 0.3, 31, endpoint=True)
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

#plot lines between significant values and medoids
for i in range(0,len(lon_c1)):
  m.plot([mlon[0],lon_c1[i]],[mlat[0],lat_c1[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c2)):
  m.plot([mlon[1],lon_c2[i]],[mlat[1],lat_c2[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c3)):
  m.plot([mlon[2],lon_c3[i]],[mlat[2],lat_c3[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c4)):
  m.plot([mlon[3],lon_c4[i]],[mlat[3],lat_c4[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c5)):
  m.plot([mlon[4],lon_c5[i]],[mlat[4],lat_c5[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c6)):
  m.plot([mlon[5],lon_c6[i]],[mlat[5],lat_c6[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c7)):
  m.plot([mlon[6],lon_c7[i]],[mlat[6],lat_c7[i]], color='0.8', linewidth=0.75, zorder=1)





mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlon, mmlat, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('1.', xy = (mlon[0],mlat[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('5.', xy = (mlon[1],mlat[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('3.', xy = (mlon[2],mlat[2]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlon[3],mlat[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('2.', xy = (mlon[4],mlat[4]), xytext=(5, 5), textcoords='offset points')
ax.annotate('4.', xy = (mlon[5],mlat[5]), xytext=(5, 5), textcoords='offset points')
ax.annotate('7.', xy = (mlon[6],mlat[6]), xytext=(5, 5), textcoords='offset points')



cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('ERA-Interim TNn, K=7, 1979-2016', fontsize=12)
cb.ax.tick_params(labelsize=8)
cb.set_label('Cluster mean silhouette coefficent', fontsize=10)
plt.savefig('/home/z5147939/hdrive/figs/tnn_son_7clust.png', bbox_inches='tight')
plt.show()




