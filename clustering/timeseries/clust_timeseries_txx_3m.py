# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset, num2date
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats

#ERA

#determine lat and lon of medoids
med = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2013_K_3_sil_0.1.nc', mode='r')
med_lon = med.variables['medoid_lon'][:]
med_lat = med.variables['medoid_lat'][:]

#bring in txx dataset
ts = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ei_3m_SON.nc', mode='r')
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






#print 'The latitude and longitude indices for n1 are %d and %d' % (lat_idx_n1, lon_idx_n1)
#print 'The latitude and longitude indices for n2 are %d and %d' % (lat_idx_n2, lon_idx_n2)
#print 'The latitude and longitude indices for n3 are %d and %d' % (lat_idx_n3, lon_idx_n3)


#define time series of medoids

n1 = ts.variables['txx'][:,lat_idx_n1,lon_idx_n1]
n2 = ts.variables['txx'][:,lat_idx_n2, lon_idx_n2]
n3 = ts.variables['txx'][:,lat_idx_n3,lon_idx_n3]





#AWAP

#determine lat and lon of medoids
med_ap = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_ap_SON_2013_K_3_sil_0.1.nc', mode='r')
med_ap_lon = med_ap.variables['medoid_lon'][:]
med_ap_lat = med_ap.variables['medoid_lat'][:]

#bring in txx dataset
ts_ap = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ap_3m_SON.nc', mode='r')
lons = ts_ap.variables['lon'][:]
lats = ts_ap.variables['lat'][:]

lat_idx_an1 = geo_idx(med_ap_lat[0], lats)
lon_idx_an1 = geo_idx(med_ap_lon[0], lons)

lat_idx_an2 = geo_idx(med_ap_lat[1], lats)
lon_idx_an2 = geo_idx(med_ap_lon[1], lons)

lat_idx_an3 = geo_idx(med_ap_lat[2], lats)
lon_idx_an3 = geo_idx(med_ap_lon[2], lons)





#print 'The latitude and longitude indices for n1 are %d and %d' % (lat_idx_an3, lon_idx_an3)
#print 'The latitude and longitude indices for n2 are %d and %d' % (lat_idx_an6, lon_idx_an6)
#print 'The latitude and longitude indices for n3 are %d and %d' % (lat_idx_an9, lon_idx_an9)

#define time series of medoids

an1 = ts_ap.variables['txx'][:,lat_idx_an1,lon_idx_an1]
an2 = ts_ap.variables['txx'][:,lat_idx_an2, lon_idx_an2]
an3 = ts_ap.variables['txx'][:,lat_idx_an3,lon_idx_an3]



#rmse and pattern correlation

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#medoid 1
rmse_n1 = rmse(n3, an2)
rmse_n1 = np.round(rmse_n1, decimals=3)

sc_n1, pv = stats.spearmanr(n3, an2, axis=0)
sc_n1 = np.round(sc_n1, decimals=3)

#medoid 2
rmse_n2 = "%2f" % rmse(n1, an1)

sc_n2, pv = stats.spearmanr(n1, an1, axis=0)
sc_n2 = "%2f" % sc_n2

#medoid 3
rmse_n3 = "%2f" % rmse(n2, an3)

sc_n3, pv = stats.spearmanr(n2, an3, axis=0)
sc_n3 = "%2f" % sc_n3




#plot figures
plt.figure(figsize=(10,4))
ax = plt.subplot(111)
ax.plot(dates, n3, label="ERA-Interim")
ax.plot(dates, an2, label="AWAP")
ax.legend(loc='lower right', prop={'size':12})
ax.text(0.34,0.01,'RMSE = %s, CORR = %s' % (rmse_n1, sc_n1), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=12)
plt.title('Medoid 1')
plt.ylabel('TXx Anomaly ($^\circ$C)')
ax.tick_params(axis='both', which='major', labelsize=8)
plt.ylim(-10,10)


# ax = plt.subplot(132)
# ax.plot(dates, n1, label="ERA-Interim")
# ax.plot(dates, an1, label="AWAP")
# ax.legend(loc='lower left', prop={'size':8})
# ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n2, sc_n2), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
# plt.title('Medoid 2', fontsize=12)
# #plt.ylabel('TXx Anomaly ($^\circ$C)')
# plt.ylim(-10,10)
# ax.tick_params(axis='both', which='major', labelsize=8)
#
# ax = plt.subplot(133)
# ax.plot(dates, n2, label="ERA-Interim")
# ax.plot(dates, an3, label="AWAP")
# ax.legend(loc='lower left', prop={'size':8})
# ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_n3, sc_n3), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=10)
# plt.title('Medoid 3', fontsize=12)
# #plt.ylabel('TXx Anomaly ($^\circ$C)', fontsize=10)
# plt.ylim(-10,10)
# ax.tick_params(axis='both', which='major', labelsize=8)
plt.savefig('/home/z5147939/hdrive/figs/clust_ts_txx_c1.png', bbox_inches='tight')
plt.show()
