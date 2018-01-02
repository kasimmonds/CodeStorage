# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset, num2date
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

#Txx
#el nino
awap_txx_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ap_3m_DJF_nino.nc', mode='r')

lon = awap_txx_e.variables['lon'][:]
lat = awap_txx_e.variables['lat'][:]
txx_ae = awap_txx_e.variables['txx'][:,:,:]
units = awap_txx_e.variables['time'].units
time_nino = awap_txx_e.variables['time'][:]

era_txx_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ei_3m_DJF_nino.nc', mode='r')
txx_ee = era_txx_e.variables['txx'][:,:,:]

diff_e = txx_ae - txx_ee
d_txx_e = np.mean((txx_ae - txx_ee), axis = 0)

d_txx_e = np.zeros((len(lat), len(lon)))
t_x = np.zeros((len(lat), len(lon)))
pv_txx_e = np.zeros((len(lat), len(lon)))
ss_txx_e = np.zeros((len(lat), len(lon)))

for i in range(0,len(lon)):
  for j in range(0,len(lat)):
    d_txx_e[j,i] = np.mean(diff_e[:,j,i], axis=0)
    t_x[j,i], pv_txx_e[j,i] = stats.ttest_1samp(diff_e[:,j,i],0.0)
    if pv_txx_e[j,i] < 0.05:
      ss_txx_e[j,i] = d_txx_e[j,i]
    else:
      ss_txx_e[j,i] = np.NaN

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_txx_e = np.round(rmse(txx_ee, txx_ae), decimals=3)




#time series
dates_nino = num2date(time_nino, units, calendar='standard') #need two dates variables as different len(time) for el nino vs la nina


sc_xe, pv = stats.spearmanr(np.ravel(txx_ee), np.ravel(txx_ae), axis=0)
sc_xe = np.round(sc_xe, decimals=3)


#la nina
awap_txx_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ap_3m_DJF_nina.nc', mode='r')
txx_al = awap_txx_l.variables['txx'][:,:,:]
time_nina = awap_txx_l.variables['time'][:]
dates_nina = num2date(time_nina, units, calendar='standard')

era_txx_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ei_3m_DJF_nina.nc', mode='r')
txx_el = era_txx_l.variables['txx'][:,:,:]


d_txx_l = np.mean((txx_al - txx_el), axis = 0)
diff_l = txx_al - txx_el
d_txx_l = np.zeros((len(lat), len(lon)))
t_x = np.zeros((len(lat), len(lon)))
pv_txx_l = np.zeros((len(lat), len(lon)))
ss_txx_l = np.zeros((len(lat), len(lon)))

for i in range(0,len(lon)):
  for j in range(0,len(lat)):
    d_txx_l[j,i] = np.mean(diff_l[:,j,i], axis=0)
    t_x[j,i], pv_txx_l[j,i] = stats.ttest_1samp(diff_l[:,j,i],0.0)
    if pv_txx_l[j,i] < 0.05:
      ss_txx_l[j,i] = d_txx_l[j,i]
    else:
      ss_txx_l[j,i] = np.NaN

rmse_txx_l = np.round(rmse(txx_el, txx_al), decimals=3)


sc_xl, pv = stats.spearmanr(np.ravel(txx_el), np.ravel(txx_al), axis=0)
sc_xl = np.round(sc_xl, decimals=3)

#TNN
#EL NINO
awap_tnn_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_3m_DJF_nino.nc', mode='r')
tnn_ae = awap_tnn_e.variables['tnn'][:,:,:]


era_tnn_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_3m_DJF_nino.nc', mode='r')
tnn_ee = era_tnn_e.variables['tnn'][:,:,:]


d_tnn_e = np.mean((tnn_ae - tnn_ee), axis=0)
diff_ne = tnn_ae - tnn_ee
d_tnn_e = np.zeros((len(lat), len(lon)))
t_x = np.zeros((len(lat), len(lon)))
pv_tnn_e = np.zeros((len(lat), len(lon)))
ss_tnn_e = np.zeros((len(lat), len(lon)))

for i in range(0,len(lon)):
  for j in range(0,len(lat)):
    d_tnn_e[j,i] = np.mean(diff_ne[:,j,i], axis=0)
    t_x[j,i], pv_tnn_e[j,i] = stats.ttest_1samp(diff_ne[:,j,i],0.0)
    if pv_tnn_e[j,i] < 0.05:
      ss_tnn_e[j,i] = d_tnn_e[j,i]
    else:
      ss_tnn_e[j,i] = np.NaN

rmse_tnn_e = np.round(rmse(tnn_ee, tnn_ae), decimals=3)

sc_ne, pv = stats.spearmanr(np.ravel(tnn_ee), np.ravel(tnn_ae), axis=0)
sc_ne = np.round(sc_ne, decimals=3)


#LA NINA
awap_tnn_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_3m_DJF_nina.nc', mode='r')
tnn_al = awap_tnn_l.variables['tnn'][:,:,:]

era_tnn_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_3m_DJF_nina.nc', mode='r')
tnn_el = era_tnn_l.variables['tnn'][:,:,:]

d_tnn_l = np.mean((tnn_al - tnn_el), axis=0)

diff_nl = tnn_al - tnn_el
d_tnn_l = np.zeros((len(lat), len(lon)))
t_x = np.zeros((len(lat), len(lon)))
pv_tnn_l = np.zeros((len(lat), len(lon)))
ss_tnn_l = np.zeros((len(lat), len(lon)))

for i in range(0,len(lon)):
  for j in range(0,len(lat)):
    d_tnn_l[j,i] = np.mean(diff_nl[:,j,i], axis=0)
    t_x[j,i], pv_tnn_l[j,i] = stats.ttest_1samp(diff_nl[:,j,i],0.0)
    if pv_tnn_l[j,i] < 0.05:
      ss_tnn_l[j,i] = d_tnn_l[j,i]
    else:
      ss_tnn_l[j,i] = np.NaN


rmse_tnn_l = np.round(rmse(tnn_el, tnn_al), decimals=3)

sc_nl, pv = stats.spearmanr(np.ravel(tnn_el), np.ravel(tnn_al), axis=0)
sc_nl = np.round(sc_nl, decimals=3)

d_txx_e = np.ma.masked_invalid(d_txx_e)
d_tnn_e = np.ma.masked_invalid(d_tnn_e)
d_txx_l = np.ma.masked_invalid(d_txx_l)
d_tnn_l = np.ma.masked_invalid(d_tnn_l)

print np.ma.max(d_txx_e), np.ma.min(d_txx_e)
print np.ma.max(d_txx_l), np.ma.min(d_txx_l)
print np.ma.max(d_tnn_e), np.ma.min(d_tnn_e)
print np.ma.max(d_tnn_l), np.ma.min(d_tnn_l)



#plot
#define global attributes
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -2., 2., 9)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)




#plot TXx during El Nino
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(223)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi,yi = m(lons,lats)
d_txx_e = np.ma.masked_invalid(d_txx_e)
ss_txx_e = np.ma.masked_invalid(ss_txx_e)
mymap= m.pcolor(xi, yi, d_txx_e, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ss_txx_e, hatch='...', norm=norm, cmap='bwr')
plt.title('TXx during El Nino', fontsize=12)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_txx_e, sc_xe), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')

#plot TNn during El Nino
ax = plt.subplot(221)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
d_tnn_e = np.ma.masked_invalid(d_tnn_e)
ss_tnn_e = np.ma.masked_invalid(ss_tnn_e)
mymap= m.pcolor(xi, yi, d_tnn_e, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_tnn_e, hatch='...', norm=norm, cmap='bwr')
plt.title('TNn during El Nino', fontsize=12)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_tnn_e, sc_ne), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')


#plot txx during la nina
ax = plt.subplot(224)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
d_txx_l = np.ma.masked_invalid(d_txx_l)
ss_txx_l = np.ma.masked_invalid(ss_txx_l)
mymap= m.pcolor(xi, yi, d_txx_l, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_txx_l, hatch='...', norm=norm, cmap='bwr')
plt.title('TXx during La Nina', fontsize=12)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_txx_l, sc_xl), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')

#plot tnn during la nina
ax = plt.subplot(222)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
d_tnn_l = np.ma.masked_invalid(d_tnn_l)
ss_tnn_l = np.ma.masked_invalid(ss_tnn_l)
mymap= m.pcolor(xi, yi, d_tnn_l, norm=norm, cmap='bwr')
ss = m.pcolor(xi, yi, ss_tnn_l, hatch='...', norm=norm, cmap='bwr')
ax.set_title('TNn during La Nina', fontsize=12)
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cb = fig.colorbar(mymap, cax, orientation='vertical')
cb.ax.tick_params(labelsize=8)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
cb.ax.tick_params(labelsize=7)
cb.set_label('AWAP - ERA-Interim', fontsize=8)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_tnn_l, sc_nl), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')
plt.savefig('/home/z5147939/hdrive/figs/era_awap_comp.png', bbox_inches='tight')
#plt.show()


#print np.max(d_txx_e), np.min(d_txx_e)
#print np.max(d_txx_l), np.min(d_txx_l)
#print np.max(d_tnn_e), np.min(d_tnn_e)
#print np.max(d_tnn_l), np.min(d_tnn_l)
