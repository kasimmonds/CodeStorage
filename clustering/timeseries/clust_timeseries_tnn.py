# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset, num2date
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats

#ERA	

#determine lat and lon of medoids
med = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_hr/tnn_SON_K_9_sil_0.1.nc', mode='r')
med_lon = med.variables['medoid_lon'][:]
med_lat = med.variables['medoid_lat'][:]

#bring in tnn dataset 
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_SON.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]
time = ts.variables['time'][:]
units = ts.variables['time'].units

dates = num2date(time, units, calendar='standard')

#determine indices for lats and lons of medoids

def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx

lat_idx_n1 = geo_idx(med_lat[0], lats)
lon_idx_n1 = geo_idx(med_lon[0], lons)

lat_idx_n2 = geo_idx(med_lat[1], lats)
lon_idx_n2 = geo_idx(med_lon[1], lons)

lat_idx_n3 = geo_idx(med_lat[2], lats)
lon_idx_n3 = geo_idx(med_lon[2], lons)

lat_idx_n4 = geo_idx(med_lat[3], lats)
lon_idx_n4 = geo_idx(med_lon[3], lons)

lat_idx_n5 = geo_idx(med_lat[4], lats)
lon_idx_n5 = geo_idx(med_lon[4], lons)

lat_idx_n6 = geo_idx(med_lat[5], lats)
lon_idx_n6 = geo_idx(med_lon[5], lons)

lat_idx_n7 = geo_idx(med_lat[6], lats)
lon_idx_n7 = geo_idx(med_lon[6], lons)

lat_idx_n8 = geo_idx(med_lat[7], lats)
lon_idx_n8 = geo_idx(med_lon[7], lons)

lat_idx_n9 = geo_idx(med_lat[8], lats)
lon_idx_n9 = geo_idx(med_lon[8], lons)


#print 'The latitude and longitude indices for n1 are %d and %d' % (lat_idx_n1, lon_idx_n1)
#print 'The latitude and longitude indices for n2 are %d and %d' % (lat_idx_n2, lon_idx_n2)
#print 'The latitude and longitude indices for n3 are %d and %d' % (lat_idx_n3, lon_idx_n3)


#define time series of medoids

n1 = ts.variables['tnn'][:,lat_idx_n1,lon_idx_n1]
n2 = ts.variables['tnn'][:,lat_idx_n2, lon_idx_n2]
n3 = ts.variables['tnn'][:,lat_idx_n3,lon_idx_n3]
n4 = ts.variables['tnn'][:,lat_idx_n4,lon_idx_n4]
n5 = ts.variables['tnn'][:,lat_idx_n5,lon_idx_n5]
n6 = ts.variables['tnn'][:,lat_idx_n6, lon_idx_n6]
n7 = ts.variables['tnn'][:,lat_idx_n7,lon_idx_n7]
n8 = ts.variables['tnn'][:,lat_idx_n8,lon_idx_n8]
n9 = ts.variables['tnn'][:,lat_idx_n9,lon_idx_n9]



#AWAP

#determine lat and lon of medoids
med_ap = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_hr/tnn_ap_SON_K_9_sil_0.1.nc', mode='r')
med_ap_lon = med_ap.variables['medoid_lon'][:]
med_ap_lat = med_ap.variables['medoid_lat'][:]

#bring in tnn dataset 
ts_ap = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_SON.nc', mode='r')
lons = ts_ap.variables['lon'][:]
lats = ts_ap.variables['lat'][:]

lat_idx_an1 = geo_idx(med_ap_lat[0], lats)
lon_idx_an1 = geo_idx(med_ap_lon[0], lons)

lat_idx_an2 = geo_idx(med_ap_lat[1], lats)
lon_idx_an2 = geo_idx(med_ap_lon[1], lons)

lat_idx_an3 = geo_idx(med_ap_lat[2], lats)
lon_idx_an3 = geo_idx(med_ap_lon[2], lons)

lat_idx_an4 = geo_idx(med_ap_lat[3], lats)
lon_idx_an4 = geo_idx(med_ap_lon[3], lons)

lat_idx_an5 = geo_idx(med_ap_lat[4], lats)
lon_idx_an5 = geo_idx(med_ap_lon[4], lons)

lat_idx_an6 = geo_idx(med_ap_lat[5], lats)
lon_idx_an6 = geo_idx(med_ap_lon[5], lons)

lat_idx_an7 = geo_idx(med_ap_lat[6], lats)
lon_idx_an7 = geo_idx(med_ap_lon[6], lons)

lat_idx_an8 = geo_idx(med_ap_lat[7], lats)
lon_idx_an8 = geo_idx(med_ap_lon[7], lons)

lat_idx_an9 = geo_idx(med_ap_lat[8], lats)
lon_idx_an9 = geo_idx(med_ap_lon[8], lons)


#print 'The latitude and longitude indices for n1 are %d and %d' % (lat_idx_an3, lon_idx_an3)
#print 'The latitude and longitude indices for n2 are %d and %d' % (lat_idx_an6, lon_idx_an6)
#print 'The latitude and longitude indices for n3 are %d and %d' % (lat_idx_an9, lon_idx_an9)

#define time series of medoids

an1 = ts_ap.variables['tnn'][:,lat_idx_an1,lon_idx_an1]
an2 = ts_ap.variables['tnn'][:,lat_idx_an2, lon_idx_an2]
an3 = ts_ap.variables['tnn'][:,lat_idx_an3,lon_idx_an3]
an4 = ts_ap.variables['tnn'][:,lat_idx_an4,lon_idx_an4]
an5 = ts_ap.variables['tnn'][:,lat_idx_an5,lon_idx_an5]
an6 = ts_ap.variables['tnn'][:,lat_idx_an6, lon_idx_an6]
an7 = ts_ap.variables['tnn'][:,lat_idx_an7,lon_idx_an7]
an8 = ts_ap.variables['tnn'][:,lat_idx_an8,lon_idx_an8]
an9 = ts_ap.variables['tnn'][:,lat_idx_an9,lon_idx_an9]

#rmse and pattern correlation

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#medoid 1
rmse_n1 = "%2f" % rmse(n1, an1)

sc_n1, pv = stats.spearmanr(n1, an1, axis=0)
sc_n1 = "%2f" % sc_n1

#medoid 2
rmse_n2 = "%2f" % rmse(n2, an2)

sc_n2, pv = stats.spearmanr(n2, an2, axis=0)
sc_n2 = "%2f" % sc_n2

#medoid 3
rmse_n3 = "%2f" % rmse(n3, an3)

sc_n3, pv = stats.spearmanr(n3, an3, axis=0)
sc_n3 = "%2f" % sc_n3

#medoid 4
rmse_n4 = "%2f" % rmse(n4, an5)

sc_n4, pv = stats.spearmanr(n4, an5, axis=0)
sc_n4 = "%2f" % sc_n4

#medoid 5
rmse_n5 = "%2f" % rmse(n5, an4)

sc_n5, pv = stats.spearmanr(n5, an4, axis=0)
sc_n5 = "%2f" % sc_n5

#medoid 6
rmse_n6 = "%2f" % rmse(n6, an7)

sc_n6, pv = stats.spearmanr(n6, an7, axis=0)
sc_n6 = "%2f" % sc_n6

#medoid 7
rmse_n7 = "%2f" % rmse(n7, an6)

sc_n7, pv = stats.spearmanr(n7, an6, axis=0)
sc_n7 = "%2f" % sc_n7

#medoid 8
rmse_n8 = "%2f" % rmse(n8, an8)

sc_n8, pv = stats.spearmanr(n8, an8, axis=0)
sc_n8 = "%2f" % sc_n8

#medoid 9
rmse_n9 = "%2f" % rmse(n9, an9)

sc_n9, pv = stats.spearmanr(n9, an9, axis=0)
sc_n9 = "%2f" % sc_n9

#plot figures 
plt.figure(figsize=(22,8))
ax = plt.subplot(331)
ax.plot(dates, n9, label="ERA-Interim")
ax.plot(dates, an9, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n9, sc_n9), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 1', fontsize=12)
plt.ylabel('TNn Anomaly ($^\circ$C)', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.ylim(-6,6)


ax = plt.subplot(332)
ax.plot(dates, n1, label="ERA-Interim")
ax.plot(dates, an1, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n1, sc_n1), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 2', fontsize=12)
#plt.ylabel('TNn Anomaly ($^\circ$C)')
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)

ax = plt.subplot(333)
ax.plot(dates, n2, label="ERA-Interim")
ax.plot(dates, an2, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n2, sc_n2), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 3', fontsize=12)
#plt.ylabel('TNn Anomaly ($^\circ$C)')
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)


ax = plt.subplot(334)
ax.plot(dates, n7, label="ERA-Interim")
ax.plot(dates, an6, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n7, sc_n7), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 4', fontsize=12)
plt.ylabel('TNn Anomaly ($^\circ$C)', fontsize=10)
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)


ax = plt.subplot(335)
ax.plot(dates, n4, label="ERA-Interim")
ax.plot(dates, an5, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n4, sc_n4), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 5', fontsize=12)
#plt.ylabel('TNn Anomaly ($^\circ$C)')
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)


ax = plt.subplot(336)
ax.plot(dates, n8, label="ERA-Interim")
ax.plot(dates, an8, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n8, sc_n8), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 6', fontsize=12)
#plt.ylabel('TNn Anomaly ($^\circ$C)')
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)


ax = plt.subplot(337)
ax.plot(dates, n5, label="ERA-Interim")
ax.plot(dates, an4, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n5, sc_n5), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 7', fontsize=12)
plt.ylabel('TNn Anomaly ($^\circ$C)', fontsize=10)
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)

ax = plt.subplot(338)
ax.plot(dates, n6, label="ERA-Interim")
ax.plot(dates, an7, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n6, sc_n6), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 8', fontsize=12)
#plt.ylabel('TNn Anomaly ($^\circ$C)')
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)


ax = plt.subplot(339)
ax.plot(dates, n3, label="ERA-Interim")
ax.plot(dates, an3, label="AWAP")
ax.legend(loc='lower left', prop={'size':8})
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n3, sc_n3), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
plt.title('Medoid 9', fontsize=12)
#plt.ylabel('TNn Anomaly ($^\circ$C)')
plt.ylim(-6,6)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.savefig('/home/z5147939/hdrive/figs/clust_ts_son_tnn_hr.png', bbox_inches='tight')
plt.show()

