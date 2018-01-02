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
print 'TNn SON'
print 'cluster 0 mean = %s' % (np.mean(sil[c1]))
print 'cluster 1 mean = %s' % (np.mean(sil[c2]))
print 'cluster 2 mean = %s' % (np.mean(sil[c3]))
print 'cluster 3 mean = %s' % (np.mean(sil[c4]))
print 'cluster 4 mean = %s' % (np.mean(sil[c5]))
print 'cluster 5 mean = %s' % (np.mean(sil[c6]))
print 'cluster 6 mean = %s' % (np.mean(sil[c7]))
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
v = np.linspace( 0, 0.35, 36, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')



#plot figure
fig = plt.figure(1, figsize=(11,8))
ax = plt.subplot(221)
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

plt.title('TNn SON', fontsize=12)




#tnn djf
nc_nd = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_7_sil_0.1.nc', mode='r')
sila = nc_nd.variables['sil_width'][:,:]
clusta = nc_nd.variables['cluster'][:,:]
#lona = nc_nd.variables['longitude'][:]
#lata = nc_nd.variables['latitude'][:]
mlona = nc_nd.variables['medoid_lon'][:]
mlata = nc_nd.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c1a = np.where(clusta == 1)
c2a = np.where(clusta == 2)
c3a = np.where(clusta == 3)
c4a = np.where(clusta == 4)
c5a = np.where(clusta == 5)
c6a = np.where(clusta == 6)
c7a = np.where(clusta == 7)


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
    if clusta[i,j] == 4:
      s_clusta[i,j] = np.mean(sila[c4a])
    if clusta[i,j] == 5:
      s_clusta[i,j] = np.mean(sila[c5a])
    if clusta[i,j] == 6:
      s_clusta[i,j] = np.mean(sila[c6a])
    if clusta[i,j] == 7:
      s_clusta[i,j] = np.mean(sila[c7a])
    if s_clusta[i,j] == 0:
      s_clusta[i,j] = np.NaN

print 'TNn DJF'
print 'cluster 0 mean = %s' % (np.mean(sila[c1a]))
print 'cluster 1 mean = %s' % (np.mean(sila[c2a]))
print 'cluster 2 mean = %s' % (np.mean(sila[c3a]))
print 'cluster 3 mean = %s' % (np.mean(sila[c4a]))
print 'cluster 4 mean = %s' % (np.mean(sila[c5a]))
print 'cluster 5 mean = %s' % (np.mean(sila[c6a]))
print 'cluster 6 mean = %s' % (np.mean(sila[c7a]))


#print stop

#make groups for clusters lat/lons
#lons
lon_c1a = []
lon_c2a = []
lon_c3a = []
lon_c4a = []
lon_c5a = []
lon_c6a = []
lon_c7a = []


#lats
lat_c1a = []
lat_c2a = []
lat_c3a = []
lat_c4a = []
lat_c5a = []
lat_c6a = []
lat_c7a = []


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


#plot figure
ax = plt.subplot(222)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi, yi = m(lons,lats)
mmlona, mmlata = m(mlona,mlata)

#large markers for significant values, small markers for insignificant
min_marker_size = 5
msize = np.zeros((sila.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if sila[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

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





mymap= m.scatter(xi, yi, s=msize, c=s_clusta, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlona, mmlata, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('1.', xy = (mlona[0],mlata[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('4.', xy = (mlona[1],mlata[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('2.', xy = (mlona[2],mlata[2]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlona[3],mlata[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('7.', xy = (mlona[4],mlata[4]), xytext=(5, 5), textcoords='offset points')
ax.annotate('5.', xy = (mlona[5],mlata[5]), xytext=(5, 5), textcoords='offset points')
ax.annotate('3.', xy = (mlona[6],mlata[6]), xytext=(5, 5), textcoords='offset points')


plt.title('TNn DJF', fontsize=12)



#txx son
nc_sx = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_7_sil_0.1.nc', mode='r')
silb = nc_sx.variables['sil_width'][:,:]
clustb = nc_sx.variables['cluster'][:,:]
#lona = nc_nd.variables['longitude'][:]
#lata = nc_nd.variables['latitude'][:]
mlonb = nc_sx.variables['medoid_lon'][:]
mlatb = nc_sx.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c1b = np.where(clustb == 1)
c2b = np.where(clustb == 2)
c3b = np.where(clustb == 3)
c4b = np.where(clustb == 4)
c5b = np.where(clustb == 5)
c6b = np.where(clustb == 6)
c7b = np.where(clustb == 7)


#for each cluster assign mean sil co value
s_clustb = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustb[i,j] == 1:
      s_clustb[i,j] = np.mean(silb[c1b])
    if clustb[i,j] == 2:
      s_clustb[i,j] = np.mean(silb[c2b])
    if clustb[i,j] == 3:
      s_clustb[i,j] = np.mean(silb[c3b])
    if clustb[i,j] == 4:
      s_clustb[i,j] = np.mean(silb[c4b])
    if clustb[i,j] == 5:
      s_clustb[i,j] = np.mean(silb[c5b])
    if clustb[i,j] == 6:
      s_clustb[i,j] = np.mean(silb[c6b])
    if clustb[i,j] == 7:
      s_clustb[i,j] = np.mean(silb[c7b])
    if s_clustb[i,j] == 0:
      s_clustb[i,j] = np.NaN

print 'TXx SON'
print 'cluster 0 mean = %s' % (np.mean(silb[c1b]))
print 'cluster 1 mean = %s' % (np.mean(silb[c2b]))
print 'cluster 2 mean = %s' % (np.mean(silb[c3b]))
print 'cluster 3 mean = %s' % (np.mean(silb[c4b]))
print 'cluster 4 mean = %s' % (np.mean(silb[c5b]))
print 'cluster 5 mean = %s' % (np.mean(silb[c6b]))
print 'cluster 6 mean = %s' % (np.mean(silb[c7b]))


#print stop

#make groups for clusters lat/lons
#lons
lon_c1b = []
lon_c2b = []
lon_c3b = []
lon_c4b = []
lon_c5b = []
lon_c6b = []
lon_c7b = []


#lats
lat_c1b = []
lat_c2b = []
lat_c3b = []
lat_c4b = []
lat_c5b = []
lat_c6b = []
lat_c7b = []


for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustb[i,j] == 1 and silb[i,j] > 0.1:
      lon_c1b.append(lon[j])
      lat_c1b.append(lat[i])
    if clustb[i,j] == 2 and silb[i,j] > 0.1:
      lon_c2b.append(lon[j])
      lat_c2b.append(lat[i])
    if clustb[i,j] == 3 and silb[i,j] > 0.1:
      lon_c3b.append(lon[j])
      lat_c3b.append(lat[i])
    if clustb[i,j] == 4 and silb[i,j] > 0.1:
      lon_c4b.append(lon[j])
      lat_c4b.append(lat[i])
    if clustb[i,j] == 5 and silb[i,j] > 0.1:
      lon_c5b.append(lon[j])
      lat_c5b.append(lat[i])
    if clustb[i,j] == 6 and silb[i,j] > 0.1:
      lon_c6b.append(lon[j])
      lat_c6b.append(lat[i])
    if clustb[i,j] == 7 and silb[i,j] > 0.1:
      lon_c7b.append(lon[j])
      lat_c7b.append(lat[i])


#plot figure
ax = plt.subplot(223)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi, yi = m(lons,lats)
mmlonb, mmlatb = m(mlonb,mlatb)

#large markers for significant values, small markers for insignificant
min_marker_size = 5
msize = np.zeros((silb.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if silb[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1b)):
  m.plot([mlonb[0],lon_c1b[i]],[mlatb[0],lat_c1b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c2b)):
  m.plot([mlonb[1],lon_c2b[i]],[mlatb[1],lat_c2b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c3b)):
  m.plot([mlonb[2],lon_c3b[i]],[mlatb[2],lat_c3b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c4b)):
  m.plot([mlonb[3],lon_c4b[i]],[mlatb[3],lat_c4b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c5b)):
  m.plot([mlonb[4],lon_c5b[i]],[mlatb[4],lat_c5b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c6b)):
  m.plot([mlonb[5],lon_c6b[i]],[mlatb[5],lat_c6b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c7b)):
  m.plot([mlonb[6],lon_c7b[i]],[mlatb[6],lat_c7b[i]], color='0.8', linewidth=0.75, zorder=1)





mymap= m.scatter(xi, yi, s=msize, c=s_clustb, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlonb, mmlatb, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('4.', xy = (mlonb[0],mlatb[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('3.', xy = (mlonb[1],mlatb[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('1.', xy = (mlonb[2],mlatb[2]), xytext=(5, 5), textcoords='offset points')
ax.annotate('5.', xy = (mlonb[3],mlatb[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('7.', xy = (mlonb[4],mlatb[4]), xytext=(5, 5), textcoords='offset points')
ax.annotate('2.', xy = (mlonb[5],mlatb[5]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlonb[6],mlatb[6]), xytext=(5, 5), textcoords='offset points')

plt.title('TXx SON', fontsize=12)


#txx djf
nc_dx = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_7_sil_0.1.nc', mode='r')
silc = nc_dx.variables['sil_width'][:,:]
clustc = nc_dx.variables['cluster'][:,:]
#lona = nc_nd.variables['longitude'][:]
#lata = nc_nd.variables['latitude'][:]
mlonc = nc_dx.variables['medoid_lon'][:]
mlatc = nc_dx.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c1c = np.where(clustb == 1)
c2c = np.where(clustb == 2)
c3c = np.where(clustb == 3)
c4c = np.where(clustb == 4)
c5c = np.where(clustb == 5)
c6c = np.where(clustb == 6)
c7c = np.where(clustb == 7)


#for each cluster assign mean sil co value
s_clustc = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustc[i,j] == 1:
      s_clustc[i,j] = np.mean(silc[c1c])
    if clustc[i,j] == 2:
      s_clustc[i,j] = np.mean(silc[c2c])
    if clustc[i,j] == 3:
      s_clustc[i,j] = np.mean(silc[c3c])
    if clustc[i,j] == 4:
      s_clustc[i,j] = np.mean(silc[c4c])
    if clustc[i,j] == 5:
      s_clustc[i,j] = np.mean(silc[c5c])
    if clustc[i,j] == 6:
      s_clustc[i,j] = np.mean(silc[c6c])
    if clustc[i,j] == 7:
      s_clustc[i,j] = np.mean(silc[c7c])
    if s_clustc[i,j] == 0:
      s_clustc[i,j] = np.NaN

print 'TXx DJF'
print 'cluster 0 mean = %s' % (np.mean(silc[c1c]))
print 'cluster 1 mean = %s' % (np.mean(silc[c2c]))
print 'cluster 2 mean = %s' % (np.mean(silc[c3c]))
print 'cluster 3 mean = %s' % (np.mean(silc[c4c]))
print 'cluster 4 mean = %s' % (np.mean(silc[c5c]))
print 'cluster 5 mean = %s' % (np.mean(silc[c6c]))
print 'cluster 6 mean = %s' % (np.mean(silc[c7c]))


#print stop

#make groups for clusters lat/lons
#lons
lon_c1c = []
lon_c2c = []
lon_c3c = []
lon_c4c = []
lon_c5c = []
lon_c6c = []
lon_c7c = []


#lats
lat_c1c = []
lat_c2c = []
lat_c3c = []
lat_c4c = []
lat_c5c = []
lat_c6c = []
lat_c7c = []


for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustc[i,j] == 1 and silc[i,j] > 0.1:
      lon_c1c.append(lon[j])
      lat_c1c.append(lat[i])
    if clustc[i,j] == 2 and silc[i,j] > 0.1:
      lon_c2c.append(lon[j])
      lat_c2c.append(lat[i])
    if clustc[i,j] == 3 and silc[i,j] > 0.1:
      lon_c3c.append(lon[j])
      lat_c3c.append(lat[i])
    if clustc[i,j] == 4 and silc[i,j] > 0.1:
      lon_c4c.append(lon[j])
      lat_c4c.append(lat[i])
    if clustc[i,j] == 5 and silc[i,j] > 0.1:
      lon_c5c.append(lon[j])
      lat_c5c.append(lat[i])
    if clustc[i,j] == 6 and silc[i,j] > 0.1:
      lon_c6c.append(lon[j])
      lat_c6c.append(lat[i])
    if clustc[i,j] == 7 and silc[i,j] > 0.1:
      lon_c7c.append(lon[j])
      lat_c7c.append(lat[i])


#plot figure
ax = plt.subplot(224)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi, yi = m(lons,lats)
mmlonc, mmlatc = m(mlonc,mlatc)

#large markers for significant values, small markers for insignificant
min_marker_size = 5
msize = np.zeros((silc.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if silc[i,j] > 0.1:
      msize[i,j] = 3 * min_marker_size
    else:
      msize[i,j] =  min_marker_size

#plot lines between significant values and medoids
for i in range(0,len(lon_c1c)):
  m.plot([mlonc[0],lon_c1c[i]],[mlatc[0],lat_c1c[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c2c)):
  m.plot([mlonc[1],lon_c2c[i]],[mlatc[1],lat_c2c[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c3c)):
  m.plot([mlonc[2],lon_c3c[i]],[mlatc[2],lat_c3c[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c4c)):
  m.plot([mlonc[3],lon_c4c[i]],[mlatc[3],lat_c4c[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c5c)):
  m.plot([mlonc[4],lon_c5c[i]],[mlatc[4],lat_c5c[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c6c)):
  m.plot([mlonc[5],lon_c6c[i]],[mlatc[5],lat_c6c[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c7c)):
  m.plot([mlonc[6],lon_c7c[i]],[mlatc[6],lat_c7c[i]], color='0.8', linewidth=0.75, zorder=1)





mymap= m.scatter(xi, yi, s=msize, c=s_clustc, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mmlonc, mmlatc, 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('2.', xy = (mlonc[0],mlatc[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('3.', xy = (mlonc[1],mlatc[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('1.', xy = (mlonc[2],mlatc[2]), xytext=(5, 5), textcoords='offset points')
ax.annotate('5.', xy = (mlonc[3],mlatc[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('7.', xy = (mlonc[4],mlatc[4]), xytext=(5, 5), textcoords='offset points')
ax.annotate('4.', xy = (mlonc[5],mlatc[5]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlonc[6],mlatc[6]), xytext=(5, 5), textcoords='offset points')
plt.title('TXx DJF', fontsize=12)
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cb = fig.colorbar(mymap, cax, orientation='vertical')
cb.ax.tick_params(labelsize=9)
cb.set_label('Cluster strength', fontsize=10)
plt.savefig('/home/z5147939/hdrive/figs/clust_7v.png', bbox_inches='tight')
plt.show()
