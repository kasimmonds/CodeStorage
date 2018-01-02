# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset, num2date
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats

#ERA	

#determine lat and lon of medoids
med = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_4_sil_0.1.nc', mode='r')
med_lon = med.variables['medoid_lon'][:]
med_lat = med.variables['medoid_lat'][:]

#bring in txx dataset 
ts = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ei_SON.nc', mode='r')
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

n1_lat = med_lat[0]
n1_lon = med_lon[0]
lat_idx_n1 = geo_idx(n1_lat, lats)
lon_idx_n1 = geo_idx(n1_lon, lons)

n2_lat = med_lat[1]
n2_lon = med_lon[1]
lat_idx_n2 = geo_idx(n2_lat, lats)
lon_idx_n2 = geo_idx(n2_lon, lons)


n3_lat = med_lat[2]
n3_lon = med_lon[2]
lat_idx_n3 = geo_idx(n3_lat, lats)
lon_idx_n3 = geo_idx(n3_lon, lons)

n4_lat = med_lat[3]
n4_lon = med_lon[3]
lat_idx_n4 = geo_idx(n4_lat, lats)
lon_idx_n4 = geo_idx(n4_lon, lons)


#print 'The latitude and longitude indices for n1 are %d and %d' % (lat_idx_n1, lon_idx_n1)
#print 'The latitude and longitude indices for n2 are %d and %d' % (lat_idx_n2, lon_idx_n2)
#print 'The latitude and longitude indices for n3 are %d and %d' % (lat_idx_n3, lon_idx_n3)


#define time series of medoids

n1 = ts.variables['txx'][:,lat_idx_n1,lon_idx_n1]
n2 = ts.variables['txx'][:,lat_idx_n2, lon_idx_n2]
n3 = ts.variables['txx'][:,lat_idx_n3,lon_idx_n3]
n4 = ts.variables['txx'][:,lat_idx_n4,lon_idx_n4]


#AWAP

#determine lat and lon of medoids
med_ap = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_4_sil_0.1.nc', mode='r')
med_ap_lon = med_ap.variables['medoid_lon'][:]
med_ap_lat = med_ap.variables['medoid_lat'][:]

#bring in txx dataset 
ts_ap = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ap_SON.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]

an1_lat = med_ap_lat[0]
an1_lon = med_ap_lon[0]
lat_idx_an1 = geo_idx(an1_lat, lats)
lon_idx_an1 = geo_idx(an1_lon, lons)

an2_lat = med_ap_lat[1]
an2_lon = med_ap_lon[1]
lat_idx_an2 = geo_idx(an2_lat, lats)
lon_idx_an2 = geo_idx(an2_lon, lons)


an3_lat = med_ap_lat[2]
an3_lon = med_ap_lon[2]
lat_idx_an3 = geo_idx(an3_lat, lats)
lon_idx_an3 = geo_idx(an3_lon, lons)

an4_lat = med_ap_lat[3]
an4_lon = med_ap_lon[3]
lat_idx_an4 = geo_idx(an4_lat, lats)
lon_idx_an4 = geo_idx(an4_lon, lons)


#print 'The latitude and longitude indices for n1 are %d and %d' % (lat_idx_n1, lon_idx_n1)
#print 'The latitude and longitude indices for n2 are %d and %d' % (lat_idx_n2, lon_idx_n2)
#print 'The latitude and longitude indices for n3 are %d and %d' % (lat_idx_n3, lon_idx_n3)


#define time series of medoids

an1 = ts_ap.variables['txx'][:,lat_idx_an1,lon_idx_an1]
an2 = ts_ap.variables['txx'][:,lat_idx_an2, lon_idx_an2]
an3 = ts_ap.variables['txx'][:,lat_idx_an3,lon_idx_an3]
an4 = ts_ap.variables['txx'][:,lat_idx_an4,lon_idx_an4]


plt.figure(1)
ax = plt.subplot(411)
ax.plot(dates, n1, label="ERA-Interim")
ax.plot(dates, an1, label="AWAP")
ax.legend(loc='lower left')
plt.title('Medoid 1')
plt.ylabel('TXx')
plt.ylim(-6,6)

ax = plt.subplot(412)
ax.plot(dates, n2, label="ERA-Interim")
ax.plot(dates, an2, label="AWAP")
ax.legend(loc='lower left')
plt.title('Medoid 2')
plt.ylabel('TXx')
plt.ylim(-6,6)

ax = plt.subplot(413)
ax.plot(dates, n3, label="ERA-Interim")
ax.plot(dates, an3, label="AWAP")
ax.legend(loc='lower left')
plt.title('Medoid 3')
plt.ylabel('TXx')
plt.ylim(-6,6)

ax = plt.subplot(414)
ax.plot(dates, n4, label="ERA-Interim")
ax.plot(dates, an4, label="AWAP")
ax.legend(loc='lower left')
plt.title('Medoid 4')
plt.ylabel('TXx')
plt.ylim(-6,6)
plt.show()

