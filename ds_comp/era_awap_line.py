# -*- coding: utf-8 -*-


#import neccessary modules
import os
from netCDF4 import Dataset, num2date
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats

#DJF
awap_djf = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ap_DJF.nc', mode='r')

lon = awap_djf.variables['lon'][:]
lat = awap_djf.variables['lat'][:]
adjf = awap_djf.variables['tnn'][0:35,:,:]
units = awap_djf.variables['time'].units
time = awap_djf.variables['time'][:]

era_djf = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ei_DJF.nc', mode='r')
edjf = era_djf.variables['tnn'][0:35,:,:]

wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180, (1,len(lon))))

ap_djf = np.zeros(adjf.shape[0]) # Preallocation
era_djf = np.zeros(adjf.shape[0])
ddjf = np.zeros(adjf.shape[0])
for i in range(adjf.shape[0]): # Don’t forget the ‘:’
  ap_djf[i] = np.sum(abs(adjf[i]) * wgtmat * ~adjf.mask[i])/np.sum(wgtmat * ~adjf.mask[i])
  era_djf[i] = np.sum(abs(edjf[i]) * wgtmat * ~edjf.mask[i])/np.sum(wgtmat * ~edjf.mask[i])
  ddjf[i] = ap_djf[i] - era_djf[i]

#MAM
awap_mam = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ap_MAM.nc', mode='r')
amam = awap_mam.variables['tnn'][:,:,:]
times = awap_mam.variables['time'][:]

dates = num2date(times, units, calendar='standard')

era_mam = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ei_MAM.nc', mode='r')
emam = era_mam.variables['tnn'][:,:,:]

ap_mam = np.zeros(amam.shape[0]) # Preallocation
era_mam = np.zeros(amam.shape[0])
dmam = np.zeros(amam.shape[0])
for i in range(amam.shape[0]): # Don’t forget the ‘:’
  ap_mam[i] = np.sum(abs(amam[i]) * wgtmat * ~amam.mask[i])/np.sum(wgtmat * ~amam.mask[i])
  era_mam[i] = np.sum(abs(emam[i]) * wgtmat * ~emam.mask[i])/np.sum(wgtmat * ~emam.mask[i])
  dmam[i] = ap_mam[i] - era_mam[i]


#JJA
awap_jja = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ap_JJA.nc', mode='r')
ajja = awap_jja.variables['tnn'][:,:,:]

era_jja = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ei_JJA.nc', mode='r')
ejja = era_jja.variables['tnn'][:,:,:]

ap_jja = np.zeros(ajja.shape[0]) # Preallocation
era_jja = np.zeros(ajja.shape[0])
djja = np.zeros(ajja.shape[0])
for i in range(ajja.shape[0]): # Don’t forget the ‘:’
  ap_jja[i] = np.sum(abs(ajja[i]) * wgtmat * ~ajja.mask[i])/np.sum(wgtmat * ~ajja.mask[i])
  era_jja[i] = np.sum(abs(ejja[i]) * wgtmat * ~ejja.mask[i])/np.sum(wgtmat * ~ejja.mask[i])
  djja[i] = ap_jja[i] - era_jja[i]

#SON
awap_son = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ap_SON.nc', mode='r')
ason = awap_son.variables['tnn'][:,:,:]

era_son = Dataset('/home/z5147939/ncfiles/comp_ds/tnn_ei_SON.nc', mode='r')
eson = era_son.variables['tnn'][:,:,:]

ap_son = np.zeros(ason.shape[0]) # Preallocation
era_son = np.zeros(ason.shape[0])
dson = np.zeros(ason.shape[0])
for i in range(ason.shape[0]): # Don’t forget the ‘:’
  ap_son[i] = np.sum(abs(ason[i]) * wgtmat * ~ason.mask[i])/np.sum(wgtmat * ~ason.mask[i])
  era_son[i] = np.sum(abs(eson[i]) * wgtmat * ~eson.mask[i])/np.sum(wgtmat * ~eson.mask[i])
  dson[i] = ap_son[i] - era_son[i]

mean_seas = (ddjf + dmam + djja + dson)/4


#DJF
awap_djft = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ap_DJF.nc', mode='r')
lon = awap_djft.variables['lon'][:]
lat = awap_djft.variables['lat'][:]
#adjf = awap_djf.variables['tnn'][0:35,:,:]
units = awap_djft.variables['time'].units
#time = awap_djf.variables['time'][:]
adjft = awap_djft.variables['txx'][0:35,:,:]




era_djft = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ei_DJF.nc', mode='r')
edjft = era_djft.variables['txx'][0:35,:,:]

wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180, (1,len(lon))))

ap_djft = np.zeros(adjft.shape[0]) # Preallocation
era_djft = np.zeros(adjft.shape[0])
ddjft = np.zeros(adjft.shape[0])
for i in range(adjft.shape[0]): # Don’t forget the ‘:’
  ap_djft[i] = np.sum(abs(adjft[i]) * wgtmat * ~adjft.mask[i])/np.sum(wgtmat * ~adjft.mask[i])
  era_djft[i] = np.sum(abs(edjft[i]) * wgtmat * ~edjft.mask[i])/np.sum(wgtmat * ~edjft.mask[i])
  ddjft[i] = ap_djft[i] - era_djft[i]

#MAM
awap_mamt = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ap_MAM.nc', mode='r')
amamt = awap_mamt.variables['txx'][:,:,:]
times = awap_mamt.variables['time'][:]
dates = num2date(times, units, calendar='standard')

era_mamt = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ei_MAM.nc', mode='r')
emamt = era_mamt.variables['txx'][:,:,:]

ap_mamt = np.zeros(amamt.shape[0]) # Preallocation
era_mamt = np.zeros(amamt.shape[0])
dmamt = np.zeros(amamt.shape[0])
for i in range(amamt.shape[0]): # Don’t forget the ‘:’
  ap_mamt[i] = np.sum(amamt[i] * wgtmat * ~amamt.mask[i])/np.sum(wgtmat * ~amamt.mask[i])
  era_mamt[i] = np.sum(emamt[i] * wgtmat * ~emamt.mask[i])/np.sum(wgtmat * ~emamt.mask[i])
  dmamt[i] = ap_mamt[i] - era_mamt[i]


#JJA
awap_jjat = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ap_JJA.nc', mode='r')
ajjat = awap_jjat.variables['txx'][:,:,:]

era_jjat = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ei_JJA.nc', mode='r')
ejjat = era_jjat.variables['txx'][:,:,:]

ap_jjat = np.zeros(ajjat.shape[0]) # Preallocation
era_jjat = np.zeros(ajjat.shape[0])
djjat = np.zeros(ajjat.shape[0])
for i in range(ajjat.shape[0]): # Don’t forget the ‘:’
  ap_jjat[i] = np.sum(ajjat[i] * wgtmat * ~ajjat.mask[i])/np.sum(wgtmat * ~ajjat.mask[i])
  era_jjat[i] = np.sum(ejjat[i] * wgtmat * ~ejjat.mask[i])/np.sum(wgtmat * ~ejjat.mask[i])
  djjat[i] = ap_jjat[i] - era_jjat[i]

#SON
awap_sont = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ap_SON.nc', mode='r')
asont = awap_sont.variables['txx'][:,:,:]

era_sont = Dataset('/home/z5147939/ncfiles/comp_ds/txx_ei_SON.nc', mode='r')
esont = era_sont.variables['txx'][:,:,:]

ap_sont = np.zeros(asont.shape[0]) # Preallocation
era_sont = np.zeros(asont.shape[0])
dsont = np.zeros(asont.shape[0])
for i in range(asont.shape[0]): # Don’t forget the ‘:’
  ap_sont[i] = np.sum(asont[i] * wgtmat * ~asont.mask[i])/np.sum(wgtmat * ~asont.mask[i])
  era_sont[i] = np.sum(esont[i] * wgtmat * ~esont.mask[i])/np.sum(wgtmat * ~esont.mask[i])
  dsont[i] = ap_sont[i] - era_sont[i]

mean_seast = (ddjft + dmamt + djjat + dsont)/4



plt.figure(1)
ax = plt.subplot(211)
ax.plot(dates, ddjf, label="DJF")
ax.plot(dates, dmam, label="MAM")
ax.plot(dates, djja, label="JJA")
ax.plot(dates, dson, label="SON")
ax.plot(dates, mean_seas, label="MEAN", linewidth=3)
ax.legend(loc='upper left')
plt.title('TNn')
plt.xlabel('Time')
plt.ylabel('AWAP - ERA')
plt.ylim(-1,1)

ax = plt.subplot(212)
ax.plot(dates, ddjft, label="DJF")
ax.plot(dates, dmamt, label="MAM")
ax.plot(dates, djjat, label="JJA")
ax.plot(dates, dsont, label="SON")
ax.plot(dates, mean_seast, label="MEAN", linewidth=3)
ax.legend(loc='upper left')
plt.title('TXx')
plt.xlabel('Time')
plt.ylabel('AWAP - ERA')
plt.ylim(-1,1)
plt.show()