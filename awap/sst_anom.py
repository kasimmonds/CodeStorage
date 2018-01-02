# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats

#ssta dataset
had_djf = Dataset('/srv/ccrc/data06/z5147939/ncfiles/sstanom/sst_anom_3m_DJF.nc', mode='r')

lon = had_djf.variables['longitude'][:]
lat = had_djf.variables['latitude'][:]
sst = had_djf.variables['sst'][:,:,:]
time = had_djf.variables['time'][:]
#time_units = had.variables['time'].units

had = Dataset('/srv/ccrc/data06/z5147939/ncfiles/sstanom/sst_anom_3m_SON.nc', mode='r')

lon = had.variables['longitude'][:]
lat = had.variables['latitude'][:]
son_sst = had.variables['sst'][:,:,:]
time = had.variables['time'][:]
#time_units = had.variables['time'].units

#bring in extra lons for shiftgrid
lon1 = had.variables['longitude'][:]
lon2 = had.variables['longitude'][:]


#bring in climate variables
#tnn
tnnd = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/tnn_fld_3m_DJF.nc', mode='r')
tnn = tnnd.variables['tnn'][:]


tnns = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/tnn_fld_3m_SON.nc', mode='r')
son_tnn = tnns.variables['tnn'][:]

#txx
txxd = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/txx_fld_3m_DJF.nc', mode='r')
txx = txxd.variables['txx'][:]

txxs = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/txx_fld_3m_SON.nc', mode='r')
son_txx = txxs.variables['txx'][:]

# #tmm
# tmmd = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/tmm_fld_3m_DJF.nc', mode='r')
# tmm = tmmd.variables['tmm'][:]
#
# tmms = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/tmm_fld_3m_SON.nc', mode='r')
# son_tmm = tmms.variables['tmm'][:]
#
# #rx1day
# rx1dayd = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/rx1day_fld_3m_DJF.nc', mode='r')
# rx1day = rx1dayd.variables['rx1day'][:]
#
# rx1days = Dataset('/srv/ccrc/data06//z5147939/ncfiles/sstanom/rx1day_fld_3m_SON.nc', mode='r')
# son_rx1day = rx1days.variables['rx1day'][:]
#
# #prcptot
# prcptotd = Dataset('/srv/ccrc/data06/z5147939//ncfiles/sstanom/prcptot_fld_3m_DJF.nc', mode='r')
# prcptot = prcptotd.variables['prcptot'][:]
#
# prcptots = Dataset('/srv/ccrc/data06/z5147939//ncfiles/sstanom/prcptot_fld_3m_SON.nc', mode='r')
# son_prcptot = prcptots.variables['prcptot'][:]



#define array of zeros for corr, pv and statistically sign for each index
#tnn
corrtnn = np.zeros((len(lat), len(lon)))
pvtnn = np.zeros((len(lat), len(lon)))
ss_tnn = np.zeros((len(lat), len(lon)))

#txx
corrtxx = np.zeros((len(lat), len(lon)))
pvtxx = np.zeros((len(lat), len(lon)))
ss_txx = np.zeros((len(lat), len(lon)))

#tmm
corrtmm = np.zeros((len(lat), len(lon)))
pvtmm = np.zeros((len(lat), len(lon)))
ss_tmm = np.zeros((len(lat), len(lon)))

#rx1day
corrx = np.zeros((len(lat), len(lon)))
pvrx = np.zeros((len(lat), len(lon)))
ss_rx = np.zeros((len(lat), len(lon)))

#prcptot
corrp = np.zeros((len(lat), len(lon)))
pvp = np.zeros((len(lat), len(lon)))
ss_p = np.zeros((len(lat), len(lon)))


#tnn
scorrtnn = np.zeros((len(lat), len(lon)))
spvtnn = np.zeros((len(lat), len(lon)))
sss_tnn = np.zeros((len(lat), len(lon)))

#txx
scorrtxx = np.zeros((len(lat), len(lon)))
spvtxx = np.zeros((len(lat), len(lon)))
sss_txx = np.zeros((len(lat), len(lon)))

#tmm
scorrtmm = np.zeros((len(lat), len(lon)))
spvtmm = np.zeros((len(lat), len(lon)))
sss_tmm = np.zeros((len(lat), len(lon)))

#rx1day
scorrx = np.zeros((len(lat), len(lon)))
spvrx = np.zeros((len(lat), len(lon)))
sss_rx = np.zeros((len(lat), len(lon)))

#prcptot
scorrp = np.zeros((len(lat), len(lon)))
spvp = np.zeros((len(lat), len(lon)))
sss_p = np.zeros((len(lat), len(lon)))




#loop through lats and lons, calculate correlations and limit for statistical significance
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      #tnn
      corrtnn[j,i], pvtnn[j,i] =  stats.spearmanr((np.squeeze(sst[:,j,i])), np.squeeze(tnn), axis=0)
      if pvtnn[j,i] < 0.05: #if stat sig
        ss_tnn[j,i] = corrtnn[j,i]
      else:
        ss_tnn[j,i] = np.NaN
      #txx
      corrtxx[j,i], pvtxx[j,i] =  stats.spearmanr((np.squeeze(sst[:,j,i])), np.squeeze(txx), axis=0)
      if pvtxx[j,i] < 0.05: #if stat sig
        ss_txx[j,i] = corrtxx[j,i]
      else:
        ss_txx[j,i] = np.NaN
      #tmm
    #   corrtmm[j,i], pvtmm[j,i] =  stats.spearmanr((np.squeeze(sst[:,j,i])), np.squeeze(tmm), axis=0)
    #   if pvtmm[j,i] < 0.05: #if stat sig
    #     ss_tmm[j,i] = corrtmm[j,i]
    #   else:
    #     ss_tmm[j,i] = np.NaN
    #   #rx1day
    #   corrx[j,i], pvrx[j,i] =  stats.spearmanr((np.squeeze(sst[:,j,i])), np.squeeze(rx1day), axis=0)
    #   if pvrx[j,i] < 0.05: #if stat sig
    #     ss_rx[j,i] = corrx[j,i]
    #   else:
    #     ss_rx[j,i] = np.NaN
    #   #prcptot
    #   corrp[j,i], pvp[j,i] =  stats.spearmanr((np.squeeze(sst[:,j,i])), np.squeeze(prcptot), axis=0)
    #   if pvp[j,i] < 0.05: #if stat sig
    #     ss_p[j,i] = corrp[j,i]
    #   else:
    #     ss_p[j,i] = np.NaN
      #tnn
      scorrtnn[j,i], spvtnn[j,i] =  stats.spearmanr((np.squeeze(son_sst[:,j,i])), np.squeeze(son_tnn), axis=0)
      if spvtnn[j,i] < 0.05: #if stat sig
        sss_tnn[j,i] = scorrtnn[j,i]
      else:
        sss_tnn[j,i] = np.NaN
      #txx
      scorrtxx[j,i], spvtxx[j,i] =  stats.spearmanr((np.squeeze(son_sst[:,j,i])), np.squeeze(son_txx), axis=0)
      if spvtxx[j,i] < 0.05: #if stat sig
        sss_txx[j,i] = scorrtxx[j,i]
      else:
        sss_txx[j,i] = np.NaN
    #   #tmm
    #   scorrtmm[j,i], spvtmm[j,i] =  stats.spearmanr((np.squeeze(son_sst[:,j,i])), np.squeeze(son_tmm), axis=0)
    #   if spvtmm[j,i] < 0.05: #if stat sig
    #     sss_tmm[j,i] = scorrtmm[j,i]
    #   else:
    #     sss_tmm[j,i] = np.NaN
    #   #rx1day
    #   scorrx[j,i], spvrx[j,i] =  stats.spearmanr((np.squeeze(son_sst[:,j,i])), np.squeeze(son_rx1day), axis=0)
    #   if spvrx[j,i] < 0.05: #if stat sig
    #     sss_rx[j,i] = scorrx[j,i]
    #   else:
    #     sss_rx[j,i] = np.NaN
    #   #prcptot
    #   scorrp[j,i], spvp[j,i] =  stats.spearmanr((np.squeeze(son_sst[:,j,i])), np.squeeze(son_prcptot), axis=0)
    #   if spvp[j,i] < 0.05: #if stat sig
    #     sss_p[j,i] = scorrp[j,i]
    #   else:
    #     sss_p[j,i] = np.NaN


#corrtxx = np.ma.masked_invalid(corrtxx)
#corrtnn = np.ma.masked_invalid(corrtnn)
#corrtmm = np.ma.masked_invalid(corrtmm)
#corrx = np.ma.masked_invalid(corrx)
#corrp = np.ma.masked_invalid(corrp)

#print np.ma.max(corrtxx), np.ma.min(corrtxx)
#print np.ma.max(corrtnn), np.ma.min(corrtnn)
#print np.ma.max(corrtmm), np.ma.min(corrtmm)
#print np.ma.max(corrx), np.ma.min(corrx)
#print np.ma.max(corrp), np.ma.min(corrp)
#print stop



#plot
#tnn

fig = plt.figure(1, figsize=(8,4))
ax = plt.subplot(2,2,2)
m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
m.drawcoastlines()
m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=8)
corrtnn, lon = shiftgrid(0., corrtnn, lon, start=True)
ss_tnn, lon = shiftgrid(0., ss_tnn, lon1, start=True)
lons, lats = np.meshgrid(lon,lat)
xi,yi = m(lons,lats)
v = np.linspace( -0.6, 0.6, 13, endpoint=True) #define colourbar ticks from -0.6 to 0.6
mymap= m.contourf(xi, yi, corrtnn, v, cmap='bwr')
ss = m.contourf(xi, yi, ss_tnn, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
plt.title('DJF TNn', fontsize=10)


ax = plt.subplot(2,2,1)
m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
m.drawcoastlines()
m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=8)
scorrtnn, lon = shiftgrid(0., scorrtnn, lon2, start=True)
sss_tnn, lon = shiftgrid(0., sss_tnn, lon2, start=True)
mymap= m.contourf(xi, yi, scorrtnn, v, cmap='bwr')
ss = m.contourf(xi, yi, sss_tnn, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
plt.title('SON TNn', fontsize=10)


#txx
ax = plt.subplot(2,2,4)
#ax10 = plt.subplot2grid((2,3), (0,1))
m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
m.drawcoastlines()
m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=8)
corrtxx, lon = shiftgrid(0., corrtxx, lon2, start=True)
ss_txx, lon = shiftgrid(0., ss_txx, lon2, start=True)
mymap= m.contourf(xi, yi, corrtxx, v, cmap='bwr')
ss = m.contourf(xi, yi, ss_txx, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
plt.title('DJF TXx', fontsize=10)



ax = plt.subplot(2,2,3)
m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
m.drawcoastlines()
m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=8)
scorrtxx, lon = shiftgrid(0., scorrtxx, lon2, start=True)
sss_txx, lon = shiftgrid(0., sss_txx, lon2, start=True)
mymap= m.contourf(xi, yi, scorrtxx, v, cmap='bwr')
ss = m.contourf(xi, yi, sss_txx, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
plt.title('SON TXx', fontsize=10)
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cb = fig.colorbar(mymap, cax, orientation='vertical')
cb.ax.tick_params(labelsize=8)
cb.set_label('Correlation', fontsize=8)
plt.savefig('/home/z5147939/hdrive/figs/sst_anom_temp.png', bbox_inches='tight')
plt.show()

# #tmm
# plt.subplot(5,2,5)
# #ax10 = plt.subplot2grid((2,3), (0,2))
# m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
# m.drawcoastlines()
# m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=6)
# m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=6)
# corrtmm, lon = shiftgrid(0., corrtmm, lon2, start=True)
# ss_tmm, lon = shiftgrid(0., ss_tmm, lon2, start=True)
# mymap= m.contourf(xi, yi, corrtmm, v, cmap='bwr')
# ss = m.contourf(xi, yi, ss_tmm, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
# cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# plt.title('DJF TMm', fontsize=10)
# cb.ax.tick_params(labelsize=7)
#
# plt.subplot(5,2,6)
# m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
# m.drawcoastlines()
# m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=6)
# m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=6)
# scorrtmm, lon = shiftgrid(0., scorrtmm, lon2, start=True)
# sss_tmm, lon = shiftgrid(0., sss_tmm, lon2, start=True)
# mymap= m.contourf(xi, yi, scorrtmm, v, cmap='bwr')
# ss = m.contourf(xi, yi, sss_tmm, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
# cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# plt.title('SON TMm', fontsize=10)
# cb.ax.tick_params(labelsize=7)
# cb.set_label('Correlation', fontsize=7)
#
#
#
#
# #rx1day
# #plt.figure(4)
# plt.subplot(5,2,7)
# #ax10 = plt.subplot2grid((2,3), (1,0))
# m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
# m.drawcoastlines()
# m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=6)
# m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=6)
# corrx, lon = shiftgrid(0., corrx, lon2, start=True)
# ss_rx, lon = shiftgrid(0., ss_rx, lon2, start=True)
# mymap= m.contourf(xi, yi, corrx, v, cmap='bwr')
# ss = m.contourf(xi, yi, ss_rx, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
# cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# plt.title('DJF RX1day', fontsize=10)
# cb.ax.tick_params(labelsize=7)
#
#
# plt.subplot(5,2,8)
# m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
# m.drawcoastlines()
# m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=6)
# m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=6)
# scorrx, lon = shiftgrid(0., scorrx, lon2, start=True)
# sss_rx, lon = shiftgrid(0., sss_rx, lon2, start=True)
# mymap= m.contourf(xi, yi, scorrx, v, cmap='bwr')
# ss = m.contourf(xi, yi, sss_rx, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
# cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# plt.title('SON RX1day', fontsize=10)
# cb.ax.tick_params(labelsize=7)
# cb.set_label('Correlation', fontsize=7)
#
#
# #prcptot
# plt.subplot(5,2,9)
# #ax10 = plt.subplot2grid((2,3), (1,2))
# m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
# m.drawcoastlines()
# m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=6)
# m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=6)
# corrp, lon = shiftgrid(0., corrp, lon2, start=True)
# ss_p, lon = shiftgrid(0., ss_p, lon2, start=True)
# mymap= m.contourf(xi, yi, corrp, v, cmap='bwr')
# ss = m.contourf(xi, yi, ss_p, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
# #cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
# cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# plt.title('DJF PRCPTOT', fontsize=10)
# cb.ax.tick_params(labelsize=6)
#
#
# plt.subplot(5,2,10)
# #ax10 = plt.subplot2grid((2,3), (1,2))
# m = Basemap(projection='cyl', llcrnrlat=-60.0, llcrnrlon=30.0, urcrnrlat=60.0, urcrnrlon=290.0)
# m.drawcoastlines()
# m.drawparallels(np.array([-60, -30, 0, 30, 60]), labels=[1,0,0,0], fontsize=6)
# m.drawmeridians(np.array([30, 95, 160, 225, 290]), labels=[0,0,0,1], fontsize=6)
# scorrp, lon = shiftgrid(0., scorrp, lon2, start=True)
# sss_p, lon = shiftgrid(0., sss_p, lon2, start=True)
# mymap= m.contourf(xi, yi, scorrp, v, cmap='bwr')
# ss = m.contourf(xi, yi, sss_p, v, hatches=['...'], cmap='bwr') #plot ss ontop of correlations
# #cb = m.colorbar(mymap,"right", size="5%", pad="2%", ticks=v)
# cb = m.colorbar(mymap,"right", size="3%", pad="2%", ticks=v)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
# plt.title('SON PRCPTOT', fontsize=10)
# cb.ax.tick_params(labelsize=6)
# cb.set_label('Correlation', fontsize=7)
