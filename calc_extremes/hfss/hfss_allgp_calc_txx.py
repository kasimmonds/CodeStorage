# -*- coding: utf-8 -*-
#import neccessary modules
import netCDF4 as nc
import numpy as np
import datetime
from matplotlib.dates import date2num
import time

#clusters
ts = nc.Dataset('/srv/ccrc/data06/z5147939/datasets/era/txx_era_1979-2016_aus.nc', mode='r')
day = ts.variables['day'][:,:,:]

#hfss
ds_hfss = nc.Dataset('/srv/ccrc/data06/z5147939/datasets/era/hfss_4pm_flux.nc', mode='r')
hfss = ds_hfss.variables['hfss'][:]
time = ds_hfss.variables['time'][:]
calendar = ds_hfss.variables['time'].calendar
units = ds_hfss.variables['time'].units
lon = ds_hfss.variables['lon'][:]
lat = ds_hfss.variables['lat'][:]


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

hfss_gp = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')


year_l = range(1979,2017) 
#year_l = range(1979,1981) 
month_l = range(1,13)
day_l = np.zeros(((456)))

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
	#for each gridpoint, i_day would need to select all days within day
	i_day = np.squeeze(np.where(days[i_year[i_month]] == day[t,i,j]))
	hfss_gp[t,i,j]  = hfss[i_year[i_month[i_day]],i,j]
	t=t+1

print hfss_gp.shape

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
f = nc.Dataset('/srv/ccrc/data06/z5147939/datasets/era/hfss_txx_gp.nc','w')

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
#hfss_c1
hfss_gp_v = f.createVariable('hfss_gp', 'f4', ('time', 'lat', 'lon',))
hfss_gp_v.long_name='Low cloud cover at tnn'
hfss_gp_v.units='Fraction'


#write data to variables
times[:] = time_axis
lons[:] = lon
lats[:] = lat
hfss_gp_v[:] = hfss_gp



#description
f.description = 'low cloud cover (ERA-Interim) at dates where Txx occurs.'
f.history = 'Created 24/08/2017'
f.source = 'Global hfss data from raijin.nci.org.au:/g/data1/ub4/erai/netcdf/6hr/atmos/oper_an_ml/v01/hfss/'
#close
f.close()




