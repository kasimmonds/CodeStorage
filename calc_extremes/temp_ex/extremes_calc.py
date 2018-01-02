# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-


#import neccessary modules
import netCDF4 as nc
import numpy as np
import datetime
from matplotlib.dates import date2num
import time


#change numbers to dates
ds_tnn = nc.Dataset('/srv/ccrc/data06/z5147939/ncfiles/ERA-Int/data/tn_era_aus_c.nc','r')
calendar = ds_tnn.variables['time'].calendar
units = ds_tnn.variables['time'].units
lon = ds_tnn.variables['lon'][:]
lat = ds_tnn.variables['lat'][:]
tn = ds_tnn.variables['mn2t'][:,:,:]
time = ds_tnn.variables['time'][:]

#change time from netcdf format to datetime
datevar = nc.num2date(time, units, calendar=calendar)
#print time, datevar
#define empty arrays for years, months, days
years = np.zeros(((len(datevar),)))
months = np.zeros(((len(datevar),)))
days = np.zeros(((len(datevar),)))
#years = np.zeros(((datevar.shape)))
#months = np.zeros(((datevar.shape)))
#days = np.zeros(((datevar.shape)))
#loop to fill empty arrays with datetime to separate each time scale
for i in range(0,len(datevar)):
  years[i] = datevar[i].year
  months[i] = datevar[i].month
  days[i] = datevar[i].day

#create empty arrays for loop
#tnn
tnn = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
#tnn = np.zeros((len(months), len(lat), len(lon)), dtype='float32')
#range of years and months so only one value per month is recorded
year_l = range(1979,2017) 
#year_l = range(1979,1981) 
month_l = range(1,13)
#month_l = range(1,4)
#variables to fill with tnn dates
day_tnn = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
month_tnn = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
year_tnn = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')

#day_tnn = np.zeros(((12*(2014-1979))), dtype=object)
#month_tnn = np.zeros(((12*(2014-1979))), dtype=object)
#year_tnn = np.zeros(((12*(2014-1979))), dtype=object)

#loop through lats, lons, years and months
for i in range(0,len(lat)):
#for i in range(0,2):
  for j in range(0,len(lon)):
  #for j in range(0,2):
    t = 0 #new time series for each gridpoint
    for y in range(0,len(year_l)):
      #indices for each year 
      i_year = np.squeeze(np.where(years == year_l[y]))
      for m in range(0,len(month_l)):
	#indices for each month
	i_month = np.squeeze(np.where(months[i_year] == month_l[m])) 
	#find index for min tn in each month (tnn)
	i_tnn  = np.squeeze(np.where(tn[i_year[i_month],i,j] == np.min(tn[i_year[i_month],i,j])))
	#account for when two minimums the same
	i_tnn = np.ravel(i_tnn)
	i_tnn = i_tnn[0]
	#print tnn.shape, tn.shape, i_tnn, i_month, i_year
	#find tnn for min tn index
	tnn[t,i,j] = tn[i_year[i_month[i_tnn]],i,j]
	#find datetime values of day, month, year indices
	day_tnn[t,i,j] = days[i_year[i_month[i_tnn]]]
	month_tnn[t,i,j] = months[i_year[i_month[i_tnn]]] 
	year_tnn[t,i,j] = years[i_year[i_month[i_tnn]]] 
	#print tnn[t,i,j]
	#print day_tnn[t,i,j], month_tnn[t,i,j], year_tnn[t,i,j], tnn[t,i,j]
	#add 1 to t so a new time is recorded at each loop
	t = t+1 


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
f = nc.Dataset('/srv/ccrc/data06/z5147939/ncfiles/ERA-Int/tnn_era_1979-2016_aus.nc','w')

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
#tnn
era_tnn = f.createVariable('tnn', 'f4', ('time', 'lat', 'lon',))
era_tnn.long_name='Monthly minimum value of daily minimum temperature (TNn)'
era_tnn.units='Celcius'

#years
years_nc = f.createVariable('year', 'f4', ('time', 'lat', 'lon',))
years_nc.long_name='Year when TNn occurred'
years_nc.units='year'

#months
months_nc = f.createVariable('month', 'f4', ('time', 'lat', 'lon',))
months_nc.long_name='Month when TNn occurred'
months_nc.units='month (1-12)'

#days
days_nc = f.createVariable('day', 'f4', ('time', 'lat', 'lon',))
days_nc.long_name='Day when TNn occurred'
days_nc.units='day of month'

#write data to variables
times[:] = time_axis
lons[:] = lon
lats[:] = lat
years_nc[:] = year_tnn
months_nc[:] = month_tnn
days_nc[:] = day_tnn
era_tnn[:] = tnn

#description
f.description = 'ERA-Interim TNn 1979-2016 and date occured over Australia'
f.history = 'Created 20/07/2017'
f.source = 'Global mn2t data from raijin.nci.org.au:/g/data1/ub4/erai/netcdf/6hr/atmos/oper_an_ml/v01/mn2t/'
#close
f.close()








