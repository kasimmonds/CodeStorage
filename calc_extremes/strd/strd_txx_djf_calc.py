# -*- coding: utf-8 -*-
#import neccessary modules
import netCDF4 as nc
import numpy as np
import datetime
from matplotlib.dates import date2num
import time

#txx DJF
#determine lat and lon of medoids
med = nc.Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_7_sil_0.1.nc', mode='r')
med_lon = med.variables['medoid_lon'][:]
med_lat = med.variables['medoid_lat'][:]

#bring in txx dataset 
ts = nc.Dataset('/srv/ccrc/data06/z5147939/ncfiles/era/txx_ei_3m_DJF.nc', mode='r')
lons = ts.variables['lon'][:]
lats = ts.variables['lat'][:]
#time = ts.variables['time'][:]
#units = ts.variables['time'].units

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

#print 'The latitude and longitude indices for n1 are %d and %d' % (lat_idx_n1, lon_idx_n1)
#print 'The latitude and longitude indices for n2 are %d and %d' % (lat_idx_n2, lon_idx_n2)
#print 'The latitude and longitude indices for n3 are %d and %d' % (lat_idx_n3, lon_idx_n3)


#define time series of medoids

#n2 = ts.variables['txx'][:,lat_idx_n1,lon_idx_n1]
#n3 = ts.variables['txx'][:,lat_idx_n2, lon_idx_n2]
#n1 = ts.variables['txx'][:,lat_idx_n3,lon_idx_n3]



#clusters
ds_txx = nc.Dataset('/srv/ccrc/data06/z5147939/datasets/era/txx_era_1979-2016_aus.nc', mode='r')
day_c2 = ds_txx.variables['day'][:,lat_idx_n1, lon_idx_n1]
day_c3 = ds_txx.variables['day'][:,lat_idx_n2, lon_idx_n2]
day_c1 = ds_txx.variables['day'][:,lat_idx_n3, lon_idx_n3]
day_c5 = ds_txx.variables['day'][:,lat_idx_n4, lon_idx_n4]
day_c7 = ds_txx.variables['day'][:,lat_idx_n5, lon_idx_n5]
day_c4 = ds_txx.variables['day'][:,lat_idx_n6, lon_idx_n6]
day_c6 = ds_txx.variables['day'][:,lat_idx_n7, lon_idx_n7]



#strd
ds_strd = nc.Dataset('/srv/ccrc/data06/z5147939/datasets/era/strd_4pm_flux.nc', mode='r')
strd = ds_strd.variables['strd'][:]
time = ds_strd.variables['time'][:]
calendar = ds_strd.variables['time'].calendar
units = ds_strd.variables['time'].units
lon = ds_strd.variables['lon'][:]
lat = ds_strd.variables['lat'][:]


#change time from netcdf format to datetime
datevar = nc.num2date(time, units, calendar=calendar)
#print time, datevar
#define empty arrays for years, months, days
years = np.zeros(((len(datevar),)))
months = np.zeros(((len(datevar),)))
days = np.zeros(((len(datevar),)))
#loop to fill empty arrays with datetime to separate each time scale
for i in range(0,len(datevar)):
  years[i] = datevar[i].year
  months[i] = datevar[i].month
  days[i] = datevar[i].day

strd_c1 = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
strd_c2 = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
strd_c3 = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
strd_c4 = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
strd_c5 = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
strd_c6 = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
strd_c7 = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')



year_l = range(1979,2017) 
#year_l = range(1979,1981) 
month_l = range(1,13)

#loop through lats, lons, years and months
for i in range(0,len(lat)):
#for i in range(0,2):
  for j in range(0,len(lon)):
  #for j in range(0,2):
    t=0
    for y in range(0,len(year_l)):
      #indices for each year 
      i_year = np.squeeze(np.where(years == year_l[y]))
      for m in range(0,len(month_l)):
	#indices for each month
	i_month = np.squeeze(np.where(months[i_year] == month_l[m]))
	i_day_c1 = np.squeeze(np.where(days[i_year[i_month]] == day_c1[t]))
	i_day_c2 = np.squeeze(np.where(days[i_year[i_month]] == day_c2[t]))
	i_day_c3 = np.squeeze(np.where(days[i_year[i_month]] == day_c3[t]))
	i_day_c4 = np.squeeze(np.where(days[i_year[i_month]] == day_c4[t]))
	i_day_c5 = np.squeeze(np.where(days[i_year[i_month]] == day_c5[t]))
	i_day_c6 = np.squeeze(np.where(days[i_year[i_month]] == day_c6[t]))
	i_day_c7 = np.squeeze(np.where(days[i_year[i_month]] == day_c7[t]))
	#print strd[i_year[i_month[i_day_c2]],i,j], day_c2[t], month_l[m], year_l[y] 
	strd_c1[t,i,j]  = strd[i_year[i_month[i_day_c1]],i,j]
	strd_c2[t,i,j]  = strd[i_year[i_month[i_day_c2]],i,j]
	strd_c3[t,i,j]  = strd[i_year[i_month[i_day_c3]],i,j]
	strd_c4[t,i,j]  = strd[i_year[i_month[i_day_c4]],i,j]
	strd_c5[t,i,j]  = strd[i_year[i_month[i_day_c5]],i,j]
	strd_c6[t,i,j]  = strd[i_year[i_month[i_day_c6]],i,j]
	strd_c7[t,i,j]  = strd[i_year[i_month[i_day_c7]],i,j]
	#print strd_c1[t,i,j], strd_c2[t,i,j], strd_c3[t,i,j]
	t=t+1

#define time axis from 1979-2013 at the 15th of each month
startyear = 1979
startmonth = 1
endyear = 2016
endmonth = 12
time_dates = np.asarray([datetime.datetime(m/12, m%12+1, 15, 0, 0) for m in xrange(startyear*12+startmonth-1, endyear*12+endmonth)])
#print time_dates
#change to numbers
time_axis = nc.date2num(time_dates, units, calendar='standard')
#print time_axis


#write nc file 
f = nc.Dataset('/srv/ccrc/data06/z5147939/datasets/era/strd_txx_djf_7clust.nc','w')

#create dimensions
f.createDimension('lon', len(lon))
f.createDimension('lat', len(lat))
f.createDimension('time', None)

#create variables

#longitude
lons = f.createVariable('lon', 'f4', ('lon',))
lons.long_name='Longitude'
lons.units = 'degrees_east'

#latitude
lats = f.createVariable('lat', 'f4', ('lat',))  
lats.long_name = 'Latitude'
lats.units = 'degrees_north'


#time
times = f.createVariable('time', 'f4', ('time',))
times.long_name = 'time'
times.units = 'hours since 1900-1-1 00:00:00' 


#create 3D variables
#strd_c1
strd_c1_v = f.createVariable('strd_c1', 'f4', ('time', 'lat', 'lon',))
strd_c1_v.long_name='strd when medoid 1 is at TXx'
strd_c1_v.units='Fraction'

#strd_c2
strd_c2_v = f.createVariable('strd_c2', 'f4', ('time', 'lat', 'lon',))
strd_c2_v.long_name='strd when medoid 2 is at TXx'
strd_c2_v.units='Fraction'

#strd_c3
strd_c3_v = f.createVariable('strd_c3', 'f4', ('time', 'lat', 'lon',))
strd_c3_v.long_name='strd when medoid 2 is at TXx'
strd_c3_v.units='Fraction'

#strd_c4
strd_c4_v = f.createVariable('strd_c4', 'f4', ('time', 'lat', 'lon',))
strd_c4_v.long_name='strd when medoid 4 is at TXx'
strd_c4_v.units='Fraction'

#strd_c5
strd_c5_v = f.createVariable('strd_c5', 'f4', ('time', 'lat', 'lon',))
strd_c5_v.long_name='strd when medoid 5 is at TXx'
strd_c5_v.units='Fraction'

#strd_c6
strd_c6_v = f.createVariable('strd_c6', 'f4', ('time', 'lat', 'lon',))
strd_c6_v.long_name='strd when medoid 6 is at TXx'
strd_c6_v.units='Fraction'


#strd_c7
strd_c7_v = f.createVariable('strd_c7', 'f4', ('time', 'lat', 'lon',))
strd_c7_v.long_name='strd when medoid 7 is at TXx'
strd_c7_v.units='Fraction'


#write data to variables
times[:] = time_axis
lons[:] = lon
lats[:] = lat
strd_c1_v[:] = strd_c1
strd_c2_v[:] = strd_c2
strd_c3_v[:] = strd_c3
strd_c4_v[:] = strd_c4
strd_c5_v[:] = strd_c5
strd_c6_v[:] = strd_c6
strd_c7_v[:] = strd_c7



#description
f.description = 'strd (ERA-Interim) at dates where medoids are at TXx in DJF, extremes not averaged.'
f.history = 'Created 24/08/2017'
f.source = 'Global strd data from raijin.nci.org.au:/g/data1/ub4/erai/netcdf/6hr/atmos/oper_an_ml/v01/strd/'
#close
f.close()




