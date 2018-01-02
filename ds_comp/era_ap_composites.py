# -*- coding: utf-8 -*-


#import neccessary modules
import os
from netCDF4 import Dataset, num2date
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats

#TNN
#el nino 
tnn_nino_ap = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ap_DJF_nino.nc', mode='r')

lon = tnn_nino_ap.variables['lon'][:]
lat = tnn_nino_ap.variables['lat'][:]
adjf = tnn_nino_ap.variables['tnn'][:,:,:]
units = tnn_nino_ap.variables['time'].units
time = tnn_nino_ap.variables['time'][:]

tnn_nino_ei = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ei_DJF_nino.nc', mode='r')
edjf = tnn_nino_ei.variables['tnn'][:,:,:]


dates = num2date(time, units, calendar='standard')


wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180, (1,len(lon))))

ap_djf = np.zeros(adjf.shape[0]) # Preallocation
era_djf = np.zeros(adjf.shape[0])
ddjf = np.zeros(adjf.shape[0])
for i in range(adjf.shape[0]): 
  ap_djf[i] = np.sum(abs(adjf[i]) * wgtmat * ~adjf.mask[i])/np.sum(wgtmat * ~adjf.mask[i])
  era_djf[i] = np.sum(abs(edjf[i]) * wgtmat * ~edjf.mask[i])/np.sum(wgtmat * ~edjf.mask[i])
  ddjf[i] = ap_djf[i] - era_djf[i]

#la nina

tnn_nina_ap = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ap_DJF_nina.nc', mode='r')

lon = tnn_nina_ap.variables['lon'][:]
lat = tnn_nina_ap.variables['lat'][:]
anina = tnn_nina_ap.variables['tnn'][:,:,:]
units = tnn_nina_ap.variables['time'].units
times = tnn_nina_ap.variables['time'][:]

era_djfl = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ei_DJF_nina.nc', mode='r')
enina = era_djfl.variables['tnn'][:,:,:]

dates_nina = num2date(times, units, calendar='standard')

wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180, (1,len(lon))))

anina_djf = np.zeros(anina.shape[0]) # Preallocation
enina_djf = np.zeros(anina.shape[0])
dnina = np.zeros(anina.shape[0])
for i in range(anina.shape[0]): 
  anina_djf[i] = np.sum(abs(anina[i]) * wgtmat * ~anina.mask[i])/np.sum(wgtmat * ~anina.mask[i])
  enina_djf[i] = np.sum(abs(enina[i]) * wgtmat * ~enina.mask[i])/np.sum(wgtmat * ~enina.mask[i])
  dnina[i] = anina_djf[i] - enina_djf[i]


#TXX
#el nino
awap_djft = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ap_DJF_nino.nc', mode='r')
lon = awap_djft.variables['lon'][:]
lat = awap_djft.variables['lat'][:]
#adjf = awap_djf.variables['tnn'][0:35,:,:]
units = awap_djft.variables['time'].units
#time = awap_djf.variables['time'][:]
adjft = awap_djft.variables['txx'][:,:,:]

era_djft = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ei_DJF_nino.nc', mode='r')
edjft = era_djft.variables['txx'][:,:,:]

wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180, (1,len(lon))))

ap_djft = np.zeros(adjft.shape[0]) # Preallocation
era_djft = np.zeros(adjft.shape[0])
ddjft = np.zeros(adjft.shape[0])
for i in range(adjft.shape[0]): # Don’t forget the ‘:’
  ap_djft[i] = np.sum(abs(adjft[i]) * wgtmat * ~adjft.mask[i])/np.sum(wgtmat * ~adjft.mask[i])
  era_djft[i] = np.sum(abs(edjft[i]) * wgtmat * ~edjft.mask[i])/np.sum(wgtmat * ~edjft.mask[i])
  ddjft[i] = ap_djft[i] - era_djft[i]

#la nina
txx_ap_ln = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ap_DJF_nina.nc', mode='r')
lon = txx_ap_ln.variables['lon'][:]
lat = txx_ap_ln.variables['lat'][:]
#adjf = awap_djf.variables['tnn'][0:35,:,:]
units = txx_ap_ln.variables['time'].units
#time = awap_djf.variables['time'][:]
tln_a = txx_ap_ln.variables['txx'][:,:,:]

txx_ei_ln = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ei_DJF_nina.nc', mode='r')
tln_e = txx_ei_ln.variables['txx'][:,:,:]

wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180, (1,len(lon))))

xap_djft = np.zeros(tln_a.shape[0]) # Preallocation
xera_djft = np.zeros(tln_a.shape[0])
xddjft = np.zeros(tln_a.shape[0])
for i in range(tln_a.shape[0]): # Don’t forget the ‘:’
  xap_djft[i] = np.sum(abs(tln_a[i]) * wgtmat * ~tln_a.mask[i])/np.sum(wgtmat * ~tln_a.mask[i])
  xera_djft[i] = np.sum(abs(tln_e[i]) * wgtmat * ~tln_e.mask[i])/np.sum(wgtmat * ~tln_e.mask[i])
  xddjft[i] = xap_djft[i] - xera_djft[i]


plt.figure(1)
ax = plt.subplot(211)
ax.plot(dates, ddjf, label="TNn")
ax.plot(dates, ddjft, label="TXx")
ax.legend(loc='upper left')
plt.title('El Nino')
plt.xlabel('Time')
plt.ylabel('AWAP - ERA')
plt.ylim(-0.5,0.5)

ax = plt.subplot(212)
ax.plot(dates_nina, dnina, label="TNn")
ax.plot(dates_nina, xddjft, label="TXx")
ax.legend(loc='upper left')
plt.title('La Nina')
plt.xlabel('Time')
plt.ylabel('AWAP - ERA')
plt.ylim(-0.5,0.5)
plt.show()