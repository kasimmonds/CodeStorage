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
v = np.linspace( 0, 0.3, 31, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
jet = plt.cm.get_cmap('jet')

#plot figure
fig = plt.figure(1, figsize=(21,10))
ax = plt.subplot2grid((4,5), (0,0))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
xi, yi = m(lons,lats)
#mmlon, mmlat = m(mlon,mlat)

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

mymap= m.scatter(xi, yi, s=msize, c=s_clust, norm=norm, cmap='rainbow', edgecolors='none', zorder=2)
medoid = m.plot(mlon[5], mlat[5], 'D', color='k', fillstyle='none', mew=2, markersize=3)
#cb = m.colorbar(mymap,"right", size="5%", pad="2%")
plt.title('TNn DJF Cluster 5', fontsize=10)
#cb.ax.tick_params(labelsize=8)
#cb.set_label('Cluster mean silhouette coefficent', fontsize=10)

#bring in tnn dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/tnn_ei_3m_DJF.nc', mode='r')
#
##determine indices for lats and lons of medoids
#
def geo_idx(dd, dd_array):
    """
      search for nearest decimal degree in an array of decimal degrees and return the index.
      np.argmin returns the indices of minium value along an axis.
      so subtract dd from all values in dd_array, take absolute value and find index of minium.
     """
    geo_idx = (np.abs(dd_array - dd)).argmin()
    return geo_idx

lat_idx_n1 = geo_idx(mlat[5], lat)
lon_idx_n1 = geo_idx(mlon[5], lon)

c1 = ts.variables['tnn'][:,lat_idx_n1,lon_idx_n1]

#bring in strd
strd = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/strd_tnn_comp_DJF_nino.nc', mode='r')
lon = strd.variables['lon'][:]
lat = strd.variables['lat'][:]
time = strd.variables['time'][:]
strd_nino = strd.variables['strd_c5'][:,:,:]

strda = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/strd_tnn_comp_DJF_nina.nc', mode='r')
strd_nina = strda.variables['strd_c5'][:,:,:]

strdn = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/strd_tnn_comp_DJF_neutral.nc', mode='r')
strd_neut = strdn.variables['strd_c5'][:,:,:]

#bring in hfls
hfls = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/hfls_tnn_comp_DJF_nino.nc', mode='r')
hfls_nino = hfls.variables['hfls_c5'][:,:,:]

hflsa = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/hfls_tnn_comp_DJF_nina.nc', mode='r')
hfls_nina = hflsa.variables['hfls_c5'][:,:,:]

hflsn = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/hfls_tnn_comp_DJF_neutral.nc', mode='r')
hfls_neut = hflsn.variables['hfls_c5'][:,:,:]


#bring in hfss
hfss = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/hfss_tnn_comp_DJF_nino.nc', mode='r')
hfss_nino = hfss.variables['hfss_c5'][:,:,:]

hfssa = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/hfss_tnn_comp_DJF_nina.nc', mode='r')
hfss_nina = hfssa.variables['hfss_c5'][:,:,:]

hfssn = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/hfss_tnn_comp_DJF_neutral.nc', mode='r')
hfss_neut = hfssn.variables['hfss_c5'][:,:,:]

#ef_c1 = hfls_c1/(hfls_c1 + hfss_c1)

#clt
clt = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/clt_tnn_comp_DJF_nino.nc', mode='r')
clt_nino = clt.variables['clt_c5'][:,:,:]

clta = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/clt_tnn_comp_DJF_nina.nc', mode='r')
clt_nina = clta.variables['clt_c5'][:,:,:]

cltn = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/clt_tnn_comp_DJF_neutral.nc', mode='r')
clt_neut = cltn.variables['clt_c5'][:,:,:]


#mslp
mslp = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/mslp_tnn_comp_DJF_nino.nc', mode='r')
lons = mslp.variables['lon'][:]
lats = mslp.variables['lat'][:]
mslp_nino = mslp.variables['mslp_c5'][:,:,:]

mslpa = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/mslp_tnn_comp_DJF_nina.nc', mode='r')
mslp_nina = mslpa.variables['mslp_c5'][:,:,:]

mslpn = Dataset('/srv/ccrc/data06/z5147939/ncfiles/7corr/mslp_tnn_comp_DJF_neutral.nc', mode='r')
mslp_neut = mslpn.variables['mslp_c5'][:,:,:]

#define empty variables
#strd
mn_strd = np.zeros((len(lat), len(lon)))
t_strd = np.zeros((len(lat), len(lon)))
pv_strd = np.zeros((len(lat), len(lon)))
ss_strd = np.zeros((len(lat), len(lon)))

mn_strd_nino = np.zeros((len(lat), len(lon)))
pv_strd_nino = np.zeros((len(lat), len(lon)))
ss_strd_nino = np.zeros((len(lat), len(lon)))

mn_strd_nina = np.zeros((len(lat), len(lon)))
pv_strd_nina = np.zeros((len(lat), len(lon)))
ss_strd_nina = np.zeros((len(lat), len(lon)))


#hfls
mn_hfls = np.zeros((len(lat), len(lon)))
t_hfls = np.zeros((len(lat), len(lon)))
pv_hfls = np.zeros((len(lat), len(lon)))
ss_hfls = np.zeros((len(lat), len(lon)))

mn_hfls_nino = np.zeros((len(lat), len(lon)))
pv_hfls_nino = np.zeros((len(lat), len(lon)))
ss_hfls_nino = np.zeros((len(lat), len(lon)))

mn_hfls_nina = np.zeros((len(lat), len(lon)))
pv_hfls_nina = np.zeros((len(lat), len(lon)))
ss_hfls_nina = np.zeros((len(lat), len(lon)))


#hfss
mn_hfss = np.zeros((len(lat), len(lon)))
t_hfss = np.zeros((len(lat), len(lon)))
pv_hfss = np.zeros((len(lat), len(lon)))
ss_hfss = np.zeros((len(lat), len(lon)))

mn_hfss_nino = np.zeros((len(lat), len(lon)))
pv_hfss_nino = np.zeros((len(lat), len(lon)))
ss_hfss_nino = np.zeros((len(lat), len(lon)))

mn_hfss_nina = np.zeros((len(lat), len(lon)))
pv_hfss_nina = np.zeros((len(lat), len(lon)))
ss_hfss_nina = np.zeros((len(lat), len(lon)))

#clt
mn_clt = np.zeros((len(lat), len(lon)))
t_clt = np.zeros((len(lat), len(lon)))
pv_clt = np.zeros((len(lat), len(lon)))
ss_clt = np.zeros((len(lat), len(lon)))

mn_clt_nino = np.zeros((len(lat), len(lon)))
pv_clt_nino = np.zeros((len(lat), len(lon)))
ss_clt_nino = np.zeros((len(lat), len(lon)))

mn_clt_nina = np.zeros((len(lat), len(lon)))
pv_clt_nina = np.zeros((len(lat), len(lon)))
ss_clt_nina = np.zeros((len(lat), len(lon)))

#mslp
mn_mslp = np.zeros((len(lats), len(lons)))
t_mslp = np.zeros((len(lats), len(lons)))
pv_mslp = np.zeros((len(lats), len(lons)))
ss_mslp = np.zeros((len(lats), len(lons)))

mn_mslp_nino = np.zeros((len(lats), len(lons)))
pv_mslp_nino = np.zeros((len(lats), len(lons)))
ss_mslp_nino = np.zeros((len(lats), len(lons)))

mn_mslp_nina = np.zeros((len(lats), len(lons)))
pv_mslp_nina = np.zeros((len(lats), len(lons)))
ss_mslp_nina = np.zeros((len(lats), len(lons)))


#loop
for i in range(0,len(lon)):
  for j in range(0,len(lat)):
    #STRD
    mn_strd[j,i] = np.mean(strd_neut[:,j,i], axis=0)
    t_strd[j,i], pv_strd[j,i] = stats.ttest_1samp(strd_neut[:,j,i],0.0)
    if pv_strd[j,i] < 0.05:
      ss_strd[j,i] = mn_strd[j,i]
    else:
      ss_strd[j,i] = np.NaN
    mn_strd_nino[j,i] = np.mean(strd_nino[:,j,i], axis=0)
    t_strd[j,i], pv_strd_nino[j,i] = stats.ttest_1samp(strd_nino[:,j,i],0.0)
    if pv_strd_nino[j,i] < 0.05:
      ss_strd_nino[j,i] = mn_strd_nino[j,i]
    else:
      ss_strd_nino[j,i] = np.NaN
    mn_strd_nina[j,i] = np.mean(strd_nina[:,j,i], axis=0)
    t_strd[j,i], pv_strd_nina[j,i] = stats.ttest_1samp(strd_nina[:,j,i],0.0)
    if pv_strd_nina[j,i] < 0.05:
      ss_strd_nina[j,i] = mn_strd_nina[j,i]
    else:
      ss_strd_nina[j,i] = np.NaN
    if clust[j,i] == 6:
        mn_hfls[j,i] = np.mean(hfls_neut[:,j,i], axis=0)
        t_hfls[j,i], pv_hfls[j,i] = stats.ttest_1samp(hfls_neut[:,j,i], 0.0, axis=0)
        if pv_hfls[j,i] < 0.05:
            ss_hfls[j,i] = mn_hfls[j,i]
        else:
            ss_hfls[j,i] = np.NaN
        mn_hfls_nino[j,i] = np.mean(hfls_nino[:,j,i], axis=0)
        t_hfls[j,i], pv_hfls_nino[j,i] = stats.ttest_1samp(hfls_nino[:,j,i], 0.0, axis=0)
        if pv_hfls_nino[j,i] < 0.05:
            ss_hfls_nino[j,i] = mn_hfls_nino[j,i]
        else:
            ss_hfls_nino[j,i] = np.NaN
        mn_hfls_nina[j,i] = np.mean(hfls_nina[:,j,i], axis=0)
        t_hfls[j,i], pv_hfls_nina[j,i] = stats.ttest_1samp(hfls_nina[:,j,i], 0.0, axis=0)
        if pv_hfls_nina[j,i] < 0.05:
            ss_hfls_nina[j,i] = mn_hfls_nina[j,i]
        else:
            ss_hfls_nina[j,i] = np.NaN
        #HFSS
        mn_hfss[j,i] = np.mean(hfss_neut[:,j,i], axis=0)
        t_hfss[j,i], pv_hfss[j,i] = stats.ttest_1samp(hfss_neut[:,j,i], 0.0, axis=0)
        if pv_hfss[j,i] < 0.05:
            ss_hfss[j,i] = mn_hfss[j,i]
        else:
            ss_hfss[j,i] = np.NaN
        mn_hfss_nino[j,i] = np.mean(hfss_nino[:,j,i], axis=0)
        t_hfss[j,i], pv_hfss_nino[j,i] = stats.ttest_1samp(hfss_nino[:,j,i], 0.0, axis=0)
        if pv_hfss_nino[j,i] < 0.05:
            ss_hfss_nino[j,i] = mn_hfss_nino[j,i]
        else:
            ss_hfss_nino[j,i] = np.NaN
        mn_hfss_nina[j,i] = np.mean(hfss_nina[:,j,i], axis=0)
        t_hfss[j,i], pv_hfss_nina[j,i] = stats.ttest_1samp(hfss_nina[:,j,i], 0.0, axis=0)
        if pv_hfss_nina[j,i] < 0.05:
            ss_hfss_nina[j,i] = mn_hfss_nina[j,i]
        else:
            ss_hfss_nina[j,i] = np.NaN
    else:
        mn_hfls[j,i] = np.NaN
        t_hfls[j,i] = np.NaN
        pv_hfls[j,i] = np.NaN
        ss_hfls[j,i] = np.NaN
        mn_hfls_nino[j,i] = np.NaN
        pv_hfls_nino[j,i] =np.NaN
        ss_hfls_nino[j,i] = np.NaN
        mn_hfls_nina[j,i] = np.NaN
        pv_hfls_nina[j,i] =np.NaN
        ss_hfls_nina[j,i] =np.NaN
        mn_hfss[j,i] = np.NaN
        t_hfss[j,i] = np.NaN
        pv_hfss[j,i] = np.NaN
        ss_hfss[j,i] = np.NaN
        mn_hfss_nino[j,i] = np.NaN
        pv_hfss_nino[j,i] = np.NaN
        ss_hfss_nino[j,i] = np.NaN
        mn_hfss_nina[j,i] = np.NaN
        pv_hfss_nina[j,i] = np.NaN
        ss_hfss_nina[j,i] = np.NaN            
    #CLT
    mn_clt[j,i] = np.mean(clt_neut[:,j,i], axis=0)
    t_clt[j,i], pv_clt[j,i] = stats.ttest_1samp(clt_neut[:,j,i], 0.0, axis=0)
    if pv_clt[j,i] < 0.05:
      ss_clt[j,i] = mn_clt[j,i]
    else:
      ss_clt[j,i] = np.NaN
    mn_clt_nino[j,i] = np.mean(clt_nino[:,j,i], axis=0)
    t_clt[j,i], pv_clt_nino[j,i] = stats.ttest_1samp(clt_nino[:,j,i], 0.0, axis=0)
    if pv_clt_nino[j,i] < 0.05:
      ss_clt_nino[j,i] = mn_clt_nino[j,i]
    else:
      ss_clt_nino[j,i] = np.NaN
    mn_clt_nina[j,i] = np.mean(clt_nina[:,j,i], axis=0)
    t_clt[j,i], pv_clt_nina[j,i] = stats.ttest_1samp(clt_nina[:,j,i], 0.0, axis=0)
    if pv_clt_nina[j,i] < 0.05:
      ss_clt_nina[j,i] = mn_clt_nina[j,i]
    else:
      ss_clt_nina[j,i] = np.NaN



for i in range(0,len(lons)):
    for j in range(0,len(lats)):
      mn_mslp[j,i] = np.mean(mslp_neut[:,j,i], axis = 0)
      t_mslp[j,i], pv_mslp[j,i] = stats.ttest_1samp(mslp_neut[:,j,i], 0.0)
      if pv_mslp[j,i] < 0.05: #if stat sig
        ss_mslp[j,i] = mn_mslp[j,i]
      else:
        ss_mslp[j,i] = np.NaN
      mn_mslp_nino[j,i] = np.mean(mslp_nino[:,j,i], axis = 0)
      t_mslp[j,i], pv_mslp_nino[j,i] = stats.ttest_1samp(mslp_nino[:,j,i], 0.0)
      if pv_mslp_nino[j,i] < 0.05: #if stat sig
        ss_mslp_nino[j,i] = mn_mslp_nino[j,i]
      else:
        ss_mslp_nino[j,i] = np.NaN
      mn_mslp_nina[j,i] = np.mean(mslp_nina[:,j,i], axis = 0)
      t_mslp[j,i], pv_mslp_nina[j,i] = stats.ttest_1samp(mslp_nina[:,j,i], 0.0)
      if pv_mslp_nina[j,i] < 0.05: #if stat sig
        ss_mslp_nina[j,i] = mn_mslp_nina[j,i]
      else:
        ss_mslp_nina[j,i] = np.NaN

#set up bounds
#v = np.linspace( -0.8, 0.8, 17, endpoint=True)
#norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
#mask oceans



#strd
ax = plt.subplot2grid((4,5), (1,0))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
lon, lat = np.meshgrid(lon,lat)
xi,yi = m(lon,lat)
mn_strd_nino = np.ma.masked_invalid(mn_strd_nino)
ss_strd_nino = np.ma.masked_invalid(ss_strd_nino)
v = np.linspace( -24, 24, 25, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
mymap= m.pcolormesh(xi, yi, mn_strd_nino, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_strd_nino, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)
plt.title('Longwave downwards', fontsize=10)
ax.set_ylabel('El Nino', labelpad=20)


ax = plt.subplot2grid((4,5), (2,0))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_strd_nina = np.ma.masked_invalid(mn_strd_nina)
ss_strd_nina = np.ma.masked_invalid(ss_strd_nina)
mymap= m.pcolormesh(xi, yi, mn_strd_nina, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_strd_nina, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)
#plt.title('Longwave downwards', fontsize=10)
ax.set_ylabel('La Nina', labelpad=20)

ax = plt.subplot2grid((4,5), (3,0))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_strd = np.ma.masked_invalid(mn_strd)
ss_strd = np.ma.masked_invalid(ss_strd)
mymap= m.pcolormesh(xi, yi, mn_strd, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_strd, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)
#plt.title('Longwave downwards', fontsize=10)
ax.set_ylabel('Neutral', labelpad=20)

#hfls
ax = plt.subplot2grid((4,5), (1,1))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_hfls_nino = np.ma.masked_invalid(mn_hfls_nino)
ss_hfls_nino = np.ma.masked_invalid(ss_hfls_nino)
v = np.linspace( -10, 10, 21, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
mymap= m.pcolormesh(xi, yi, mn_hfls_nino, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_hfls_nino, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)
plt.title('Latent heat flux', fontsize=10)

ax = plt.subplot2grid((4,5), (2,1))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_hfls_nina = np.ma.masked_invalid(mn_hfls_nina)
ss_hfls_nina = np.ma.masked_invalid(ss_hfls_nina)
mymap= m.pcolormesh(xi, yi, mn_hfls_nina, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_hfls_nina, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)
#plt.title('Latent heat flux', fontsize=10)


ax = plt.subplot2grid((4,5), (3,1))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_hfls = np.ma.masked_invalid(mn_hfls)
ss_hfls = np.ma.masked_invalid(ss_hfls)
mymap= m.pcolormesh(xi, yi, mn_hfls, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_hfls, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)
#plt.title('Latent heat flux', fontsize=10)

#hfss
ax = plt.subplot2grid((4,5), (1,2))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_hfss_nino = np.ma.masked_invalid(mn_hfss_nino)
ss_hfss_nino = np.ma.masked_invalid(ss_hfss_nino)
v = np.linspace( -10, 10, 21, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
mymap= m.pcolormesh(xi, yi, mn_hfss_nino, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_hfss_nino, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)
plt.title('Sensible heat flux', fontsize=10)

ax = plt.subplot2grid((4,5), (2,2))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_hfss_nina = np.ma.masked_invalid(mn_hfss_nina)
ss_hfss_nina = np.ma.masked_invalid(ss_hfss_nina)
mymap= m.pcolormesh(xi, yi, mn_hfss_nina, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_hfss_nina, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('$W/m^2$', fontsize=8)


ax = plt.subplot2grid((4,5), (3,2))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_hfss = np.ma.masked_invalid(mn_hfss)
ss_hfss = np.ma.masked_invalid(ss_hfss)
mymap= m.pcolormesh(xi, yi, mn_hfss, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_hfss, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('W/m2', fontsize=8)


#clt
ax = plt.subplot2grid((4,5), (1,3))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_clt_nino = np.ma.masked_invalid(mn_clt_nino)
ss_clt_nino = np.ma.masked_invalid(ss_clt_nino)
v = np.linspace( -0.26, 0.26, 27, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
mymap= m.pcolormesh(xi, yi, mn_clt_nino, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_clt_nino, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.ax.tick_params(labelsize=6)
cb.set_label('Fraction', fontsize=8)
plt.title('Total cloud cover', fontsize=10)

ax = plt.subplot2grid((4,5), (2,3))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_clt_nina = np.ma.masked_invalid(mn_clt_nina)
ss_clt_nina = np.ma.masked_invalid(ss_clt_nina)
mymap= m.pcolormesh(xi, yi, mn_clt_nina, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_clt_nina, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.ax.tick_params(labelsize=6)
cb.set_label('Fraction', fontsize=8)

ax = plt.subplot2grid((4,5), (3,3))
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=7)
mn_clt = np.ma.masked_invalid(mn_clt)
ss_clt = np.ma.masked_invalid(ss_clt)
mymap= m.pcolormesh(xi, yi, mn_clt, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_clt, hatch='...', norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.ax.tick_params(labelsize=6)
cb.set_label('Fraction', fontsize=8)
#plt.title('Total cloud cover', fontsize=10)

#MSLP
ax = plt.subplot2grid((4,5), (1,4))
m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=80.0, urcrnrlat=20.0, urcrnrlon=200.0)
m.drawcoastlines()
m.drawparallels(np.array([-60, -40, -20, 0, 20]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([80, 110, 140, 170, 200]), labels=[0,0,0,1], fontsize=7)
lons, lats = np.meshgrid(lons,lats)
xi,yi = m(lons,lats)
mn_mslp_nino = np.ma.masked_invalid(mn_mslp_nino)
ss_mslp_nino = np.ma.masked_invalid(ss_mslp_nino)
v = np.linspace( -540, 540, 28, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
mymap= m.pcolormesh(xi, yi, mn_mslp_nino, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ss_mslp_nino, hatch='...', norm=norm, cmap=plt.cm.bwr)
cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('Pa', fontsize=8)
plt.title('Mean sea level pressure', fontsize=10)

ax = plt.subplot2grid((4,5), (2,4))
m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=80.0, urcrnrlat=20.0, urcrnrlon=200.0)
m.drawcoastlines()
m.drawparallels(np.array([-60, -40, -20, 0, 20]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([80, 110, 140, 170, 200]), labels=[0,0,0,1], fontsize=7)
mn_mslp_nina = np.ma.masked_invalid(mn_mslp_nina)
ss_mslp_nina = np.ma.masked_invalid(ss_mslp_nina)
mymap= m.pcolormesh(xi, yi, mn_mslp_nina, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ss_mslp_nina, hatch='...', norm=norm, cmap=plt.cm.bwr)
cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('Pa', fontsize=8)

ax = plt.subplot2grid((4,5), (3,4))
m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=80.0, urcrnrlat=20.0, urcrnrlon=200.0)
m.drawcoastlines()
m.drawparallels(np.array([-60, -40, -20, 0, 20]), labels=[1,0,0,0], fontsize=7)
m.drawmeridians(np.array([80, 110, 140, 170, 200]), labels=[0,0,0,1], fontsize=7)
mn_mslp = np.ma.masked_invalid(mn_mslp)
ss_mslp = np.ma.masked_invalid(ss_mslp)
mymap= m.pcolormesh(xi, yi, mn_mslp, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ss_mslp, hatch='...', norm=norm, cmap=plt.cm.bwr)
cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
cb.ax.tick_params(labelsize=6)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.set_label('Pa', fontsize=8)
#plt.title('Mean sea level pressure', fontsize=10)

print "STRD"
print "NEUTRAL", np.ma.min(mn_strd), np.ma.max(mn_strd)
print "EL NINO", np.ma.min(mn_strd_nino), np.ma.max(mn_strd_nino)
print "LA NINA", np.ma.min(mn_strd_nina), np.ma.max(mn_strd_nina)

print "HFLS"
print "NEUTRAL", np.ma.min(mn_hfls), np.ma.max(mn_hfls)
print "EL NINO", np.ma.min(mn_hfls_nino), np.ma.max(mn_hfls_nino)
print "LA NINA", np.ma.min(mn_hfls_nina), np.ma.max(mn_hfls_nina)

print "HFSS"
print "NEUTRAL", np.ma.min(mn_hfss), np.ma.max(mn_hfss)
print "EL NINO", np.ma.min(mn_hfss_nino), np.ma.max(mn_hfss_nino)
print "LA NINA", np.ma.min(mn_hfss_nina), np.ma.max(mn_hfss_nina)

print "CLT"
print "NEUTRAL", np.ma.min(mn_clt), np.ma.max(mn_clt)
print "EL NINO", np.ma.min(mn_clt_nino), np.ma.max(mn_clt_nino)
print "LA NINA", np.ma.min(mn_clt_nina), np.ma.max(mn_clt_nina)

print "MSLP"
print "NEUTRAL", np.ma.min(mn_mslp), np.ma.max(mn_mslp)
print "EL NINO", np.ma.min(mn_mslp_nino), np.ma.max(mn_mslp_nino)
print "LA NINA", np.ma.min(mn_mslp_nina), np.ma.max(mn_mslp_nina)
plt.savefig('/home/z5147939/hdrive/figs/tnn_djf_c5_enso.png', bbox_inches='tight')
#plt.show()
