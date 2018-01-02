# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
import matplotlib.colors as colors

############################
###TNN SON 
############################
nce = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_7_sil_0.1.nc', mode='r')
sil = nce.variables['sil_width'][:,:]
clust = nce.variables['cluster'][:,:]
lon = nce.variables['longitude'][:]
lat = nce.variables['latitude'][:]
mlon = nce.variables['medoid_lon'][:]
mlat = nce.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c4 = np.where(clust == 4)
c5 = np.where(clust == 5)
c6 = np.where(clust == 6)
c7 = np.where(clust == 7)


#for each cluster assign mean sil co value
s_clust = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
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


#make groups for clusters lat/lons
#lons
lon_c4 = []
lon_c5 = []
lon_c6 = []
lon_c7 = []


#lats
#lat_c1 = []
#lat_c2 = []
#lat_c3 = []
lat_c4 = []
lat_c5 = []
lat_c6 = []
lat_c7 = []


for i in range(0,len(lat)):
  for j in range(0,len(lon)):
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
plt.figure(1, figsize=(15,6))
ax = plt.subplot(122)
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

for i in range(0,len(lon_c4)):
  m.plot([mlon[3],lon_c4[i]],[mlat[3],lat_c4[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c5)):
  m.plot([mlon[4],lon_c5[i]],[mlat[4],lat_c5[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c6)):
  m.plot([mlon[5],lon_c6[i]],[mlat[5],lat_c6[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c7)):
  m.plot([mlon[6],lon_c7[i]],[mlat[6],lat_c7[i]], color='0.8', linewidth=0.75, zorder=1)





mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlon[3], mlat[3], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlon[4], mlat[4], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlon[5], mlat[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlon[6], mlat[6], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('6.', xy = (mlon[3],mlat[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('2.', xy = (mlon[4],mlat[4]), xytext=(5, 5), textcoords='offset points')
ax.annotate('4.', xy = (mlon[5],mlat[5]), xytext=(5, 5), textcoords='offset points')
ax.annotate('7.', xy = (mlon[6],mlat[6]), xytext=(5, 5), textcoords='offset points')



cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('TNn SON', fontsize=12)
cb.ax.tick_params(labelsize=8)
cb.set_label('Cluster mean silhouette coefficent', fontsize=10)

###########################
#txx son 
###########################
nc_sx = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_7_sil_0.1.nc', mode='r')
silb = nc_sx.variables['sil_width'][:,:]
clustb = nc_sx.variables['cluster'][:,:]
mlonb = nc_sx.variables['medoid_lon'][:]
mlatb = nc_sx.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c4b = np.where(clustb == 4)
c5b = np.where(clustb == 5)
c6b = np.where(clustb == 6)
c7b = np.where(clustb == 7)


#for each cluster assign mean sil co value
s_clustb = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
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




#print stop

#make groups for clusters lat/lons
#lons
lon_c4b = []
lon_c5b = []
lon_c6b = []
lon_c7b = []


#lats
lat_c4b = []
lat_c5b = []
lat_c6b = []
lat_c7b = []


for i in range(0,len(lat)):
  for j in range(0,len(lon)):
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
ax = plt.subplot(121)
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

for i in range(0,len(lon_c4b)):
  m.plot([mlonb[3],lon_c4b[i]],[mlatb[3],lat_c4b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c5b)):
  m.plot([mlonb[4],lon_c5b[i]],[mlatb[4],lat_c5b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c6b)):
  m.plot([mlonb[5],lon_c6b[i]],[mlatb[5],lat_c6b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c7b)):
  m.plot([mlonb[6],lon_c7b[i]],[mlatb[6],lat_c7b[i]], color='0.8', linewidth=0.75, zorder=1)





mymap= m.scatter(xi, yi, s=msize, c=s_clustb, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlonb[3], mlatb[3], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonb[4], mlatb[4], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonb[5], mlatb[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonb[6], mlatb[6], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('5.', xy = (mlonb[3],mlatb[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('7.', xy = (mlonb[4],mlatb[4]), xytext=(5, 5), textcoords='offset points')
ax.annotate('2.', xy = (mlonb[5],mlatb[5]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlonb[6],mlatb[6]), xytext=(5, 5), textcoords='offset points')



cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('TXx SON', fontsize=12)
cb.ax.tick_params(labelsize=8)
#cb.set_label('Cluster mean silhouette coefficent', fontsize=10)

plt.savefig('/home/z5147939/hdrive/figs/clust_7_son.png', bbox_inches='tight')
plt.show()

#######################
#tnn djf 
########################
fig2 = plt.figure(2, figsize=(15,6)) 
nc_nd = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_7_sil_0.1.nc', mode='r')
sila = nc_nd.variables['sil_width'][:,:]
clusta = nc_nd.variables['cluster'][:,:]
mlona = nc_nd.variables['medoid_lon'][:]
mlata = nc_nd.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c1a = np.where(clusta == 1)
c2a = np.where(clusta == 2)
c4a = np.where(clusta == 4)
c6a = np.where(clusta == 6)



#for each cluster assign mean sil co value
s_clusta = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clusta[i,j] == 1:
      s_clusta[i,j] = np.mean(sila[c1a])
    if clusta[i,j] == 2:
      s_clusta[i,j] = np.mean(sila[c2a])
    if clusta[i,j] == 4:
      s_clusta[i,j] = np.mean(sila[c4a])
    if clusta[i,j] == 6:
      s_clusta[i,j] = np.mean(sila[c6a])
    if s_clusta[i,j] == 0:
      s_clusta[i,j] = np.NaN




#print stop

#make groups for clusters lat/lons
#lons
lon_c1a = []
lon_c2a = []
lon_c4a = []
lon_c6a = []


#lats
lat_c1a = []
lat_c2a = []
lat_c4a = []
lat_c6a = []



for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clusta[i,j] == 1 and sila[i,j] > 0.1:
      lon_c1a.append(lon[j])
      lat_c1a.append(lat[i])
    if clusta[i,j] == 2 and sila[i,j] > 0.1:
      lon_c2a.append(lon[j])
      lat_c2a.append(lat[i])
    if clusta[i,j] == 4 and sila[i,j] > 0.1:
      lon_c4a.append(lon[j])
      lat_c4a.append(lat[i])
    if clusta[i,j] == 6 and sila[i,j] > 0.1:
      lon_c6a.append(lon[j])
      lat_c6a.append(lat[i])



#plot figure
plt.figure(2, figsize=(15,6))
ax = plt.subplot(122)
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


for i in range(0,len(lon_c4a)):
  m.plot([mlona[3],lon_c4a[i]],[mlata[3],lat_c4a[i]], color='0.8', linewidth=0.75, zorder=1)


for i in range(0,len(lon_c6a)):
  m.plot([mlona[5],lon_c6a[i]],[mlata[5],lat_c6a[i]], color='0.8', linewidth=0.75, zorder=1)






mymap= m.scatter(xi, yi, s=msize, c=s_clusta, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlona[0], mlata[0], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlona[1], mmlata[1], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlona[3], mmlata[3], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlona[5], mmlata[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('1.', xy = (mlona[0],mlata[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('4.', xy = (mlona[1],mlata[1]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlona[3],mlata[3]), xytext=(5, 5), textcoords='offset points')
ax.annotate('5.', xy = (mlona[5],mlata[5]), xytext=(5, 5), textcoords='offset points')




cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('TNn DJF', fontsize=12)
cb.ax.tick_params(labelsize=8)
cb.set_label('Cluster mean silhouette coefficent', fontsize=10)

###############################
#txx son 
###############################
nc_sx = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_7_sil_0.1.nc', mode='r')
silb = nc_sx.variables['sil_width'][:,:]
clustb = nc_sx.variables['cluster'][:,:]
mlonb = nc_sx.variables['medoid_lon'][:]
mlatb = nc_sx.variables['medoid_lat'][:]

#seperate each cluster to determine mean silhouette coefficient
c1b = np.where(clustb == 1)
c6b = np.where(clustb == 6)
c7b = np.where(clustb == 7)


#for each cluster assign mean sil co value
s_clustb = np.zeros((clust.shape))
for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustb[i,j] == 1:
      s_clustb[i,j] = np.mean(silb[c1b])
    if clustb[i,j] == 6:
      s_clustb[i,j] = np.mean(silb[c6b])
    if clustb[i,j] == 7:
      s_clustb[i,j] = np.mean(silb[c7b])
    if s_clustb[i,j] == 0:
      s_clustb[i,j] = np.NaN



#print stop

#make groups for clusters lat/lons
#lons
lon_c1b = []
lon_c6b = []
lon_c7b = []


#lats
lat_c1b = []
lat_c6b = []
lat_c7b = []


for i in range(0,len(lat)):
  for j in range(0,len(lon)):
    if clustb[i,j] == 1 and silb[i,j] > 0.1:
      lon_c1b.append(lon[j])
      lat_c1b.append(lat[i])
    if clustb[i,j] == 6 and silb[i,j] > 0.1:
      lon_c6b.append(lon[j])
      lat_c6b.append(lat[i])
    if clustb[i,j] == 7 and silb[i,j] > 0.1:
      lon_c7b.append(lon[j])
      lat_c7b.append(lat[i])


#plot figure
ax = plt.subplot(121)
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


for i in range(0,len(lon_c6b)):
  m.plot([mlonb[5],lon_c6b[i]],[mlatb[5],lat_c6b[i]], color='0.8', linewidth=0.75, zorder=1)

for i in range(0,len(lon_c7b)):
  m.plot([mlonb[6],lon_c7b[i]],[mlatb[6],lat_c7b[i]], color='0.8', linewidth=0.75, zorder=1)





mymap= m.scatter(xi, yi, s=msize, c=s_clustb, norm=norm, cmap=jet, edgecolors='none', zorder=2)
medoid = m.plot(mlonb[0], mlatb[0], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonb[5], mlatb[5], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)
medoid = m.plot(mlonb[6], mlatb[6], 'D', color='k', fillstyle='none', mew=1.5, markersize=3)

#annotate medoids according to strength
ax.annotate('4.', xy = (mlonb[0],mlatb[0]), xytext=(5, 5), textcoords='offset points')
ax.annotate('2.', xy = (mlonb[5],mlatb[5]), xytext=(5, 5), textcoords='offset points')
ax.annotate('6.', xy = (mlonb[6],mlatb[6]), xytext=(5, 5), textcoords='offset points')



cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('TXx DJF', fontsize=12)
cb.ax.tick_params(labelsize=8)
plt.savefig('/home/z5147939/hdrive/figs/clust_7_djf.png', bbox_inches='tight')
plt.show()




