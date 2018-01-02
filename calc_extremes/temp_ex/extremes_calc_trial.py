# -*- coding: utf-8 -*-


#import neccessary modules
import netCDF4 as nc
import numpy as np
import datetime
from matplotlib.dates import date2num
import decimal


#change numbers to dates
ds_tnn = nc.Dataset('/home/z5147939/ncfiles/comp_ds/ERA-Int_TN_1979-2013_aus.nc','r')
calendar = ds_tnn.variables['time'].calendar
units = ds_tnn.variables['time'].units
lon = ds_tnn.variables['longitude'][:]
lat = ds_tnn.variables['latitude'][:]
tn = ds_tnn.variables['t2m'][:,:,:]
time = ds_tnn.variables['time'][:]

#change time from netcdf format to datetime
datevar = nc.num2date(time, units, calendar=calendar)

#define empty arrays for years, months, days
years = np.zeros(((len(datevar),)), dtype=object)
months = np.zeros(((len(datevar),)), dtype=object)
days = np.zeros(((len(datevar),)), dtype=object)
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
tnn = np.zeros(((12*(2014-1979)), len(lat), len(lon)), dtype='float32')
print tnn.shape
#range of years and months so only one value per month is recorded
year_l = range(1979,2014) 
#year_l = range(1979,1981) 
month_l = range(1,13)
#month_l = range(1,4)
#variables to fill with tnn dates
day_tnn = np.zeros(((12*(2014-1979)), len(lat), len(lon)), dtype='float32')
month_tnn = np.zeros(((12*(2014-1979)), len(lat), len(lon)), dtype='float32')
year_tnn = np.zeros(((12*(2014-1979)), len(lat), len(lon)), dtype='float32')

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
	#i_tnn  = np.zeros((1,))
	#print np.squeeze(np.where(tn[i_year[i_month],i,j] == np.min(tn[i_year[i_month],i,j]))).shape, i_tnn[0].shape
	i_tnn= np.squeeze(np.where(tn[i_year[i_month],i,j] == np.min(tn[i_year[i_month],i,j])))
	i_tnn = np.ravel(i_tnn)
	#print len(i_tnn)
	i_tnn = i_tnn[0]
	#print i_tnn.dtype
        #print isinstance(i_tnn, decimal.Decimal)
	#i_tnn = i_tnn[0]
	"""
	if i_tnn[0,0] == i_tnn[0,1]:
	  i_tnn = i_tnn[0,0]
	else:
	  i_tnn = i_tnn[0,0]
	"""
	#print len(np.min(tn[i_year[i_month],i,j]))
	#if len(i_tnn) > 1:
	  #i_tnn = np.min(i_tnn)
	#print i_tnn.dtype
	#print i, j, y, m
	#print i_year[i_month[i_tnn]]
	#find tnn for min tn index
	#print tn[i_year[i_month[i_tnn]],i,j], tnn[t,i,j]
	#print list(tn[i_year[i_month[i_tnn]],i,j]).count(tn[i_year[i_month[i_tnn]],i,j][0])
	tnn[t,i,j] = tn[i_year[i_month[i_tnn]],i,j]
	#find datetime values of day, month, year indices
	day_tnn[t,i,j] = days[i_year[i_month[i_tnn]]]
	month_tnn[t,i,j] = months[i_year[i_month[i_tnn]]] 
	year_tnn[t,i,j] = years[i_year[i_month[i_tnn]]] 
	#print tnn[t,i,j]
	#print day_tnn[t,i,j], month_tnn[t,i,j], year_tnn[t,i,j], tnn[t,i,j]
	#add 1 to t so a new time is recorded at each loop
	t = t+1 

#try change from object to float - doesnt work for 3D
#tnn = tnn.astype(np.float)
#day_tnn = np.vstack(day_tnn[:,:,:]).astype(np.float)
#month_tnn = np.vstack(month_tnn[:,:,:]).astype(np.float)
#year_tnn = np.vstack(year_tnn[:,:,:]).astype(np.float)


#define time axis from 1979-2013 at the 15th of each month
startyear = 1979
startmonth = 1
endyear = 2013
endmonth = 12
time_dates = [datetime.date(m/12, m%12+1, 15) for m in xrange(startyear*12+startmonth-1, endyear*12+endmonth)]
#change to numbers
time_axis = date2num(time_dates)
#print time_axis


#write nc file 
f = nc.Dataset('tnn_ERA_trial.nc','w')

#create dimensions
f.createDimension('lon', len(lon))
f.createDimension('lat', len(lat))
f.createDimension('time', None)

#create variables
#time
times = f.createVariable('time', 'f4', ('time',))
times.units = 'hours since 0001-01-01 00:00:00'  

#longitude
lons = f.createVariable('lon', 'f4', ('lon',))
lons.units = 'degrees_east'
lons.long_name='Longitude'

#latitude
lats = f.createVariable('lat', 'f4', ('lat',))  
lats.units = 'degrees_north'
lats.long_name='Latitude'

#create 3D variables
#tnn
era_tnn = f.createVariable('tnn', 'f4', ('time', 'lat', 'lon',))
era_tnn.units='Kelvin'
era_tnn.long_name='ERA-Interium TNn'

#years
years_nc = f.createVariable('years', 'f4', ('time', 'lat', 'lon',))
years_nc.long_name='Years'

#months
months_nc = f.createVariable('months', 'f4', ('time', 'lat', 'lon',))
months_nc.long_name='Months'

#days
days_nc = f.createVariable('days', 'f4', ('time', 'lat', 'lon',))
days_nc.long_name='Days'

#write data to variables
times[:] = time_axis
lons[:] = lon
lats[:] = lat
years_nc[:] = year_tnn
months_nc[:] = month_tnn
days_nc[:] = day_tnn
era_tnn[:] = tnn

#description
f.description = "Trial TNn data from ERA-Interim"   

#close
f.close()








