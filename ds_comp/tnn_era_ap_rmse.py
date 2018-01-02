# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats
import pandas as pd
import matplotlib.colors as colors

#TNN DIFFERENCES
#AWAP
#DJF
awap_djf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_3m_DJF.nc', mode='r')

lon = awap_djf.variables['lon'][:]
lat = awap_djf.variables['lat'][:]
adjf = awap_djf.variables['tnn'][:,:,:]
time_units = awap_djf.variables['time'].units
time = awap_djf.variables['time'][:]

era_djf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_3m_DJF.nc', mode='r')
edjf = era_djf.variables['tnn'][:,:,:]

ndjf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/nino_era_3m_DJF.nc', mode='r')
nino_djf = ndjf.variables['sst'][:]

#set up variables to be created

#correlations and pvalues for both era and awap
acorrdjf = np.zeros((len(lat), len(lon)))
apvdjf = np.zeros((len(lat), len(lon)))
ecorrdjf = np.zeros((len(lat), len(lon)))
epvdjf = np.zeros((len(lat), len(lon)))

#statistically significant correlations for era and awap
ess_djf = np.zeros((len(lat), len(lon)))
ass_djf = np.zeros((len(lat), len(lon)))

#difference between era and awap correlations
ddjf = np.zeros((len(lat), len(lon)))

#print nino_djf.shape, edjf.shape
#print stop

#calculate in one loop
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      ecorrdjf[j,i], epvdjf[j,i] =  stats.spearmanr((np.squeeze(edjf[:,j,i])), np.squeeze(nino_djf), axis=0)
      if epvdjf[j,i] < 0.05:
        ess_djf[j,i] = ecorrdjf[j,i]
      else:
        ess_djf[j,i] = np.NaN
      acorrdjf[j,i], apvdjf[j,i] =  stats.spearmanr((np.squeeze(adjf[:,j,i])), np.squeeze(nino_djf), axis=0)
      if apvdjf[j,i] < 0.05:
        ass_djf[j,i] = acorrdjf[j,i]
      else:
        ass_djf[j,i] = np.NaN
      ddjf[j,i] = acorrdjf[j,i] - ecorrdjf[j,i]

ecorrdjf = ecorrdjf[~np.isnan(ecorrdjf)]
acorrdjf = acorrdjf[~np.isnan(acorrdjf)]

rmse_djf = np.round(np.sqrt(np.nanmean((ecorrdjf.astype('float64') - acorrdjf.astype('float64'))**2)), decimals=3)
sc_djf, pv = np.round(stats.spearmanr(np.ravel(ecorrdjf), np.ravel(acorrdjf), axis=0), decimals=3)
sc_djf = "%.3f" % sc_djf




##MAM
#awap_mam = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_MAM.nc', mode='r')
#amam = awap_mam.variables['tnn'][:,:,:]
#times = awap_mam.variables['time'][:]

#era_mam = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_MAM.nc', mode='r')
#emam = era_mam.variables['tnn'][:,:,:]

#nmam = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/nino_era_MAM.nc', mode='r')
#nino_mam = nmam.variables['sst'][:]


##write the loops for correlation with nino34, then take difference between awap and era-interim
##define array of zeros

##set up variables to be created

##correlations and pvalues for both era and awap
#acorrmam = np.zeros((len(lat), len(lon)))
#apvmam = np.zeros((len(lat), len(lon)))
#ecorrmam = np.zeros((len(lat), len(lon)))
#epvmam = np.zeros((len(lat), len(lon)))

##statistically significant correlations for era and awap
#ess_mam = np.zeros((len(lat), len(lon)))
#ass_mam = np.zeros((len(lat), len(lon)))

##difference between era and awap correlations
#dmam = np.zeros((len(lat), len(lon)))

##calculate in one loop
#for i in range(0,len(lon)):
    #for j in range(0,len(lat)):
      #ecorrmam[j,i], epvmam[j,i] =  stats.spearmanr((np.squeeze(emam[:,j,i])), np.squeeze(nino_mam), axis=0)
      #if epvmam[j,i] < 0.05:
        #ess_mam[j,i] = ecorrmam[j,i]
      #else:
        #ess_mam[j,i] = np.NaN
      #acorrmam[j,i], apvmam[j,i] =  stats.spearmanr((np.squeeze(amam[:,j,i])), np.squeeze(nino_mam), axis=0)
      #if apvmam[j,i] < 0.05:
        #ass_mam[j,i] = acorrmam[j,i]
      #else:
        #ass_mam[j,i] = np.NaN
      #dmam[j,i] = acorrmam[j,i] - ecorrmam[j,i]


##calculate rmse
#rmse_mam2 = np.nanmean((ecorrmam.astype('float64') - acorrmam.astype('float64'))**2)
#rmse_mam = "%2f" % np.sqrt(rmse_mam2)

##calculate pattern correlation
#sc_mam, pv = stats.spearmanr(np.ravel(ecorrmam), np.ravel(acorrmam), axis=0)
#sc_mam = "%2f" % sc_mam


##JJA
#awap_jja = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_JJA.nc', mode='r')
#ajja = awap_jja.variables['tnn'][:,:,:]


#era_jja = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_JJA.nc', mode='r')
#ejja = era_jja.variables['tnn'][:,:,:]

#njja = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/nino_era_JJA.nc', mode='r')
#nino_jja = njja.variables['sst'][:]


##write the loop for correlation w/ nino3.4 and difference
##define array of zeros

##correlations and pvalues for both era and awap
#acorrjja = np.zeros((len(lat), len(lon)))
#apvjja = np.zeros((len(lat), len(lon)))
#ecorrjja = np.zeros((len(lat), len(lon)))
#epvjja = np.zeros((len(lat), len(lon)))

##statistically significant correlations for era and awap
#ess_jja = np.zeros((len(lat), len(lon)))
#ass_jja = np.zeros((len(lat), len(lon)))

##difference between era and awap correlations
#djja = np.zeros((len(lat), len(lon)))

##calculate in one loop
#for i in range(0,len(lon)):
    #for j in range(0,len(lat)):
      #ecorrjja[j,i], epvjja[j,i] =  stats.spearmanr((np.squeeze(ejja[:,j,i])), np.squeeze(nino_jja), axis=0)
      #if epvjja[j,i] < 0.05:
        #ess_jja[j,i] = ecorrjja[j,i]
      #else:
        #ess_jja[j,i] = np.NaN
      #acorrjja[j,i], apvjja[j,i] =  stats.spearmanr((np.squeeze(ajja[:,j,i])), np.squeeze(nino_jja), axis=0)
      #if apvjja[j,i] < 0.05:
        #ass_jja[j,i] = acorrjja[j,i]
      #else:
        #ass_jja[j,i] = np.NaN
      #djja[j,i] = acorrjja[j,i] - ecorrjja[j,i]



##calculate rmse
#rmse_jja2 = np.round(np.sqrt(np.nanmean((ecorrjja.astype('float64') - acorrjja.astype('float64'))**2)), decimals=3)


##calculate corr pattern
#sc_jja, pv = np.round(stats.spearmanr(np.ravel(ecorrjja), np.ravel(acorrjja), axis=0), decimals=3)



#SON
awap_son = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_3m_SON.nc', mode='r')
ason = awap_son.variables['tnn'][:,:,:]

era_son = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_3m_SON.nc', mode='r')
eson = era_son.variables['tnn'][:,:,:]

nson = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/nino_era_3m_SON.nc', mode='r')
nino_son = nson.variables['sst'][:]


#write the loops + limit for statistical significance
#define array of zeros

#correlations and pvalues for both era and awap
acorrson = np.zeros((len(lat), len(lon)))
apvson = np.zeros((len(lat), len(lon)))
ecorrson = np.zeros((len(lat), len(lon)))
epvson = np.zeros((len(lat), len(lon)))

#statistically significant correlations for era and awap
ess_son = np.zeros((len(lat), len(lon)))
ass_son = np.zeros((len(lat), len(lon)))

#difference between era and awap correlations
dson = np.zeros((len(lat), len(lon)))

#calculate in one loop
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      ecorrson[j,i], epvson[j,i] =  stats.spearmanr((np.squeeze(eson[:,j,i])), np.squeeze(nino_son), axis=0)
      if epvson[j,i] < 0.05:
        ess_son[j,i] = ecorrson[j,i]
      else:
        ess_son[j,i] = np.NaN
      acorrson[j,i], apvson[j,i] =  stats.spearmanr((np.squeeze(ason[:,j,i])), np.squeeze(nino_son), axis=0)
      if apvson[j,i] < 0.05:
        ass_son[j,i] = acorrson[j,i]
      else:
        ass_son[j,i] = np.NaN
      dson[j,i] = acorrson[j,i] - ecorrson[j,i]


#calculate rmse
rmse_son = np.round(np.sqrt(np.nanmean((ecorrson.astype('float64') - acorrson.astype('float64'))**2)), decimals=3)
#rmse_son = "%2f" % np.sqrt(rmse_son2)

#calculate corr pattern
sc_son, pv = np.round(stats.spearmanr(np.ravel(ecorrson), np.ravel(acorrson), axis=0), decimals=3)
sc_son = "%.3f" % sc_son





#plot

#define global attributes
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.6, 0.6, 13)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)

#plot DJF comp
plt.figure(figsize=(12,5))
ax = plt.subplot(2,3,1)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
xi,yi = m(lons,lats)
ddjf = np.ma.masked_invalid(ddjf)
print np.ma.max(ddjf), np.ma.min(ddjf)
mymap= m.pcolormesh(xi, yi, ddjf, norm=norm, cmap='bwr')
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
plt.title('DJF', fontsize=12)
cb.ax.tick_params(labelsize=6)
cb.set_label('AWAP - ERA-Interim', fontsize=8)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_djf, sc_djf), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')

#plot DJF awap
plt.subplot(2,3,2)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
xi,yi = m(lons,lats)
acorrdjf = np.ma.masked_invalid(acorrdjf)
ass_djf = np.ma.masked_invalid(ass_djf)
print np.ma.max(acorrdjf), np.ma.min(acorrdjf)
mymap= m.pcolor(xi, yi, acorrdjf, norm=norm,  cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ass_djf, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
plt.title('DJF', fontsize=12)
cb.ax.tick_params(labelsize=6)
cb.set_label('Corr(AWAP, NINO3.4)', fontsize=8)

#plot DJF era
plt.subplot(2,3,3)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
xi,yi = m(lons,lats)
ecorrdjf = np.ma.masked_invalid(ecorrdjf)
ess_djf = np.ma.masked_invalid(ess_djf)
print np.ma.max(ecorrdjf), np.ma.min(ecorrdjf)
mymap= m.pcolor(xi, yi, ecorrdjf, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ess_djf, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
plt.title('DJF', fontsize=12)
cb.ax.tick_params(labelsize=6)
cb.set_label('Corr(ERA-Interim, NINO3.4)', fontsize=8)

##plot MAM comp
#ax = plt.subplot(4,3,4)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
#dmam = np.ma.masked_invalid(dmam)
#mymap= m.pcolormesh(xi, yi, dmam, norm=norm, cmap=plt.cm.bwr)
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#plt.title('MAM', fontsize=12)
#cb.ax.tick_params(labelsize=6)
#cb.set_label('AWAP - ERA-Interim', fontsize=8)
#ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_mam, sc_mam), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=7, fontweight='bold')

##plot MAM awap
#ax = plt.subplot(4,3,5)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
#acorrmam = np.ma.masked_invalid(acorrmam)
#ass_mam = np.ma.masked_invalid(ass_mam)
#mymap= m.pcolor(xi, yi, acorrmam, norm=norm, cmap=plt.cm.bwr)
#ss = m.pcolor(xi, yi, ass_mam, hatch='...', norm=norm, cmap=plt.cm.bwr)
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#plt.title('MAM', fontsize=12)
#cb.ax.tick_params(labelsize=6)
#cb.set_label('Corr(AWAP, NINO3.4)', fontsize=8)

##plot MAM era
#ax = plt.subplot(4,3,6)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
#ecorrmam = np.ma.masked_invalid(ecorrmam)
#ess_mam = np.ma.masked_invalid(ess_mam)
#mymap= m.pcolor(xi, yi, ecorrmam, norm=norm, cmap=plt.cm.bwr)
#ss = m.pcolor(xi, yi, ess_mam, hatch='...', norm=norm, cmap=plt.cm.bwr)
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#plt.title('MAM', fontsize=12)
#cb.ax.tick_params(labelsize=6)
#cb.set_label('Corr(ERA-Interim, NINO3.4)', fontsize=8)

##plot JJA comp
#ax = plt.subplot(4,3,7)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
#djja = np.ma.masked_invalid(djja)
#mymap= m.pcolormesh(xi, yi, djja, norm=norm, cmap=plt.cm.bwr)
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#plt.title('JJA', fontsize=12)
#cb.ax.tick_params(labelsize=6)
#cb.set_label('AWAP - ERA-Interim', fontsize =8)
#ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_jja, sc_jja), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=7, fontweight='bold')

##plot JJA awap
#ax = plt.subplot(4,3,8)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
#acorrjja = np.ma.masked_invalid(acorrjja)
#ass_jja = np.ma.masked_invalid(ass_jja)
#mymap= m.pcolor(xi, yi, acorrjja, norm=norm, cmap=plt.cm.bwr)
#ss = m.pcolor(xi, yi, ass_jja, hatch='...', norm=norm, cmap=plt.cm.bwr)
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#plt.title('JJA', fontsize=12)
#cb.ax.tick_params(labelsize=6)
#cb.set_label('Corr(AWAP, NINO3.4)', fontsize=8)

##plot JJA era
#ax = plt.subplot(4,3,9)
#m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
#m.drawcoastlines()
#m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
#m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
#ecorrjja = np.ma.masked_invalid(ecorrjja)
#ess_jja = np.ma.masked_invalid(ess_jja)
#mymap= m.pcolor(xi, yi, ecorrjja, norm=norm, cmap=plt.cm.bwr)
#ss = m.pcolor(xi, yi, ess_jja, hatch='...', norm=norm, cmap=plt.cm.bwr)
#cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
#plt.title('JJA', fontsize=12)
#cb.ax.tick_params(labelsize=6)
#cb.set_label('Corr(ERA-Interim, NINO3.4)', fontsize=8)

#plot SON comp
ax = plt.subplot(2,3,4)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
dson = np.ma.masked_invalid(dson)
print np.ma.max(dson), np.ma.max(dson)
mymap= m.pcolormesh(xi, yi, dson, norm=norm, cmap=plt.cm.bwr)
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
plt.title('SON', fontsize=12)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
cb.ax.tick_params(labelsize=6)
cb.set_label('AWAP - ERA-Interim', fontsize=8)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_son, sc_son), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')

#plot SON awap
ax = plt.subplot(2,3,5)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
acorrson = np.ma.masked_invalid(acorrson)
ass_son = np.ma.masked_invalid(ass_son)
print np.ma.max(acorrson), np.ma.min(acorrson)
mymap= m.pcolor(xi, yi, acorrson, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ass_son, hatch='...', norm=norm, cmap=plt.cm.bwr)
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
plt.title('SON', fontsize=12)
cb.ax.tick_params(labelsize=6)
cb.set_label('Corr(AWAP, NINO3.4)', fontsize=8)

#plot SON era
ax = plt.subplot(2,3,6)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
ecorrson = np.ma.masked_invalid(ecorrson)
ess_son = np.ma.masked_invalid(ess_son)
print np.ma.min(ecorrson), np.ma.max(ecorrson)
mymap= m.pcolor(xi, yi, ecorrson, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ess_son, hatch='...', norm=norm, cmap=plt.cm.bwr)
cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
for label in cb.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
plt.title('SON', fontsize=12)
cb.ax.tick_params(labelsize=6)
cb.set_label('Corr(ERA-Interim, NINO3.4)', fontsize=8)
plt.savefig('/home/z5147939/hdrive/figs/tnn_era_awap_n34.png', bbox_inches='tight')
plt.show()

#print 'djf min is %s and max is %s' % (np.nanmin(ddjf), np.nanmax(ddjf))
#print 'mam min is %s and max is %s' % (np.nanmin(dmam), np.nanmax(dmam))
#print 'jja min is %s and max is %s' % (np.nanmin(djja), np.nanmax(djja))
#print 'son min is %s and max is %s' % (np.nanmin(dson), np.nanmax(dson))
