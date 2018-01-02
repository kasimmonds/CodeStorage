# -*- coding: utf-8 -*-


#import neccessary modules
import netCDF4 as nc
import numpy as np
import datetime
from matplotlib.dates import date2num
import time


#bring in tx data
ds_tx = nc.Dataset('/srv/ccrc/data06/z5147939/ncfiles/ERA-Int/data/tx_era_aus_c.nc','r')
calendar = ds_tx.variables['time'].calendar
units = ds_tx.variables['time'].units
lon = ds_tx.variables['lon'][:]
lat = ds_tx.variables['lat'][:]
tx = ds_tx.variables['mx2t'][:,:,:]
time = ds_tx.variables['time'][:]

#change time from netcdf format to datetime
datevar = nc.num2date(time, units, calendar=calendar)

#define empty arrays for years, months, days
years = np.zeros(((len(datevar),)))
months = np.zeros(((len(datevar),)))
days = np.zeros(((len(datevar),)))
#loop to fill empty arrays with datetime to separate each time scale
for i in range(0,len(datevar)):
  years[i] = datevar[i].year
  months[i] = datevar[i].month
  days[i] = datevar[i].day

#create empty arrays for loop
#txx
txx = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
#range of years and months so only one value per month is recorded
year_l = range(1979,2017) 
month_l = range(1,13)
#variables to fill with txx dates
day_txx = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
month_txx = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')
year_txx = np.zeros(((12*(2017-1979)), len(lat), len(lon)), dtype='float32')


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
	#find index for min tx in each month (txx)
	i_txx  = np.squeeze(np.where(tx[i_year[i_month],i,j] == np.max(tx[i_year[i_month],i,j])))
	#account for when two maximums the same
	i_txx = np.ravel(i_txx)
	i_txx = i_txx[0]
	#find txx for min tx index
	txx[t,i,j] = tx[i_year[i_month[i_txx]],i,j]
	#find datetime values of day, month, year indices
	day_txx[t,i,j] = days[i_year[i_month[i_txx]]]
	month_txx[t,i,j] = months[i_year[i_month[i_txx]]] 
	year_txx[t,i,j] = years[i_year[i_month[i_txx]]] 
	#add 1 to t so a new time is recorded at each loop
	t = t+1 


#define time axis from 1979-2013 at the 15th of each month
startyear = 1979
startmonth = 1
endyear = 2016
endmonth = 12
time_dates = np.asarray([datetime.datetime(m/12, m%12+1, 15, 0, 0) for m in xrange(startyear*12+startmonth-1, endyear*12+endmonth)])
#change to numbers
time_axis = nc.date2num(time_dates, units, calendar='standard')
#print time_axis


#write nc file 
f = nc.Dataset('/srv/ccrc/data06/z5147939/ncfiles/ERA-Int/txx_era_1979-2016_aus.nc','w')

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
#txx
era_txx = f.createVariable('txx', 'f4', ('time', 'lat', 'lon',))
era_txx.long_name='Monthly maximum value of daily maximum temperature (TNn)'
era_txx.units='Celcius'

#years
years_nc = f.createVariable('year', 'f4', ('time', 'lat', 'lon',))
years_nc.long_name='Year when TXx occurred'
years_nc.units='year'

#months
months_nc = f.createVariable('month', 'f4', ('time', 'lat', 'lon',))
months_nc.long_name='Month when TXx occurred'
months_nc.units='month (1-12)'

#days
days_nc = f.createVariable('day', 'f4', ('time', 'lat', 'lon',))
days_nc.long_name='Day when TXx occurred'
days_nc.units='day of month'

#write data to variables
times[:] = time_axis
lons[:] = lon
lats[:] = lat
years_nc[:] = year_txx
months_nc[:] = month_txx
days_nc[:] = day_txx
era_txx[:] = txx

#description
f.description = 'ERA-Interim TXx 1979-2016 and date occured over Australia'
f.history = 'Created 20/07/2017'
f.source = 'Global mx2t data from raijin.nci.org.au:/g/data1/ub4/erai/netcdf/6hr/atmos/oper_an_ml/v01/mx2t/'
#close
f.close()








