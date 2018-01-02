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

jra_djf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_jra_3m_DJF.nc', mode='r')
jdjf = jra_djf.variables['tnn'][:,:,:]

ndjf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/nino_era_3m_DJF.nc', mode='r')
nino_djf = ndjf.variables['sst'][:]

#set up variables to be created

#correlations and pvalues for both era and awap
acorrdjf = np.zeros((len(lat), len(lon)))
apvdjf = np.zeros((len(lat), len(lon)))

ecorrdjf = np.zeros((len(lat), len(lon)))
epvdjf = np.zeros((len(lat), len(lon)))

jcorrdjf = np.zeros((len(lat), len(lon)))
jpvdjf = np.zeros((len(lat), len(lon)))


#statistically significant correlations for era and awap
ess_djf = np.zeros((len(lat), len(lon)))
ass_djf = np.zeros((len(lat), len(lon)))
jss_djf = np.zeros((len(lat), len(lon)))


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
      jcorrdjf[j,i], jpvdjf[j,i] =  stats.spearmanr((np.squeeze(jdjf[:,j,i])), np.squeeze(nino_djf), axis=0)
      if jpvdjf[j,i] < 0.05:
        jss_djf[j,i] = jcorrdjf[j,i]
      else:
        jss_djf[j,i] = np.NaN

ecordjf = ecorrdjf[~np.isnan(ecorrdjf)]
acordjf = acorrdjf[~np.isnan(acorrdjf)]
jcordjf = jcorrdjf[~np.isnan(jcorrdjf)]

rmse_djf = np.round(np.sqrt(np.nanmean((ecordjf.astype('float64') - acordjf.astype('float64'))**2)), decimals=3)
rmse_jdjf = np.round(np.sqrt(np.nanmean((jcordjf.astype('float64') - acordjf.astype('float64'))**2)), decimals=3)

sc_djf, pv = np.round(stats.spearmanr(np.ravel(ecordjf), np.ravel(acordjf), axis=0), decimals=3)
sc_djf = "%.3f" % sc_djf

sc_jdjf, pv = np.round(stats.spearmanr(np.ravel(jcordjf), np.ravel(acordjf), axis=0), decimals=3)
sc_jdjf = "%.3f" % sc_jdjf

#SON
awap_son = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_3m_SON.nc', mode='r')
ason = awap_son.variables['tnn'][:,:,:]

era_son = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_3m_SON.nc', mode='r')
eson = era_son.variables['tnn'][:,:,:]

jra_son = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_jra_3m_SON.nc', mode='r')
json = jra_son.variables['tnn'][:,:,:]

nson = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/nino_era_3m_SON.nc', mode='r')
nino_son = nson.variables['sst'][:]


#write the loops + limit for statistical significance
#define array of zeros

#correlations and pvalues for both era and awap
acorrson = np.zeros((len(lat), len(lon)))
apvson = np.zeros((len(lat), len(lon)))
ecorrson = np.zeros((len(lat), len(lon)))
epvson = np.zeros((len(lat), len(lon)))
jcorrson = np.zeros((len(lat), len(lon)))
jpvson = np.zeros((len(lat), len(lon)))

#statistically significant correlations for era and awap
ess_son = np.zeros((len(lat), len(lon)))
ass_son = np.zeros((len(lat), len(lon)))
jss_son = np.zeros((len(lat), len(lon)))



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
      jcorrson[j,i], jpvson[j,i] =  stats.spearmanr((np.squeeze(json[:,j,i])), np.squeeze(nino_son), axis=0)
      if jpvson[j,i] < 0.05:
        jss_son[j,i] = jcorrson[j,i]
      else:
        jss_son[j,i] = np.NaN

ecorson = ecorrson[~np.isnan(ecorrson)]
acorson = acorrson[~np.isnan(acorrson)]
jcorson = jcorrson[~np.isnan(jcorrson)]
#calculate rmse
rmse_son = np.round(np.sqrt(np.nanmean((ecorson.astype('float64') - acorson.astype('float64'))**2)), decimals=3)
rmse_json = np.round(np.sqrt(np.nanmean((jcorson.astype('float64') - acorson.astype('float64'))**2)), decimals=3)
#calculate corr pattern
sc_son, pv = np.round(stats.spearmanr(np.ravel(ecorson), np.ravel(acorson), axis=0), decimals=3)
sc_son = "%.3f" % sc_son

sc_json, pv = np.round(stats.spearmanr(np.ravel(jcorson), np.ravel(acorson), axis=0), decimals=3)
sc_json = "%.3f" % sc_json



#plot

#define global attributes
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( -0.6, 0.6, 13)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)

#plot son awap
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(231)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
xi,yi = m(lons,lats)
acorrson = np.ma.masked_invalid(acorrson)
ass_son = np.ma.masked_invalid(ass_son)
mymap= m.pcolor(xi, yi, acorrson, norm=norm,  cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ass_son, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
plt.title('AWAP', fontsize=12)
ax.set_ylabel('SON', fontsize=12, labelpad=20)

#plot son era
ax = plt.subplot(232)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
xi,yi = m(lons,lats)
ecorrson = np.ma.masked_invalid(ecorrson)
ess_son = np.ma.masked_invalid(ess_son)
mymap= m.pcolor(xi, yi, ecorrson, norm=norm,  cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ess_son, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
plt.title('ERA-Interim', fontsize=12)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_son, sc_son), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')


#plot son jra
ax = plt.subplot(233)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
xi,yi = m(lons,lats)
jcorrson = np.ma.masked_invalid(jcorrson)
jss_son = np.ma.masked_invalid(jss_son)
mymap= m.pcolor(xi, yi, jcorrson, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, jss_son, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
plt.title('JRA-55', fontsize=12)
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_json, sc_json), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')

#plot djf awap
ax = plt.subplot(234)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
acorrdjf = np.ma.masked_invalid(acorrdjf)
ass_djf = np.ma.masked_invalid(ass_djf)
mymap= m.pcolor(xi, yi, acorrdjf, norm=norm,  cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ass_djf, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
ax.set_ylabel('DJF', fontsize=12, labelpad=20)

#plot djf era
ax = plt.subplot(235)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
ecorrdjf = np.ma.masked_invalid(ecorrdjf)
ess_djf = np.ma.masked_invalid(ess_djf)
mymap= m.pcolor(xi, yi, ecorrdjf, norm=norm,  cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, ess_djf, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_djf, sc_djf), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')

#plot djf jra
ax = plt.subplot(236)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=6)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=6)
jcorrdjf = np.ma.masked_invalid(jcorrdjf)
jss_djf = np.ma.masked_invalid(jss_djf)
mymap= m.pcolor(xi, yi, jcorrdjf, norm=norm, cmap=plt.cm.bwr)
ss = m.pcolor(xi, yi, jss_djf, hatch='...', norm=norm, cmap=plt.cm.bwr) #plot ss ontop of correlations
ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_jdjf, sc_jdjf), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cb = fig.colorbar(mymap, cax, orientation='vertical')
cb.ax.tick_params(labelsize=8)
cb.set_label('Correlation', fontsize=8)
plt.savefig('/home/z5147939/hdrive/figs/tnn_comp_n34.png', bbox_inches='tight')
plt.show()

#print 'djf min is %s and max is %s' % (np.nanmin(ddjf), np.nanmax(ddjf))
#print 'mam min is %s and max is %s' % (np.nanmin(dmam), np.nanmax(dmam))
#print 'jja min is %s and max is %s' % (np.nanmin(djja), np.nanmax(djja))
#print 'son min is %s and max is %s' % (np.nanmin(dson), np.nanmax(dson))
