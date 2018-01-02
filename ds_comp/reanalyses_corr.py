# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset, num2date
import numpy as np
import matplotlib
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter

#DJF
#TNn
awap_tnn_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_3m_DJF.nc', mode='r')

lon = awap_tnn_e.variables['lon'][:]
lat = awap_tnn_e.variables['lat'][:]
tnn_ae = awap_tnn_e.variables['tnn'][:,:,:]
units = awap_tnn_e.variables['time'].units
time_nino = awap_tnn_e.variables['time'][:]

era_tnn_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_3m_DJF.nc', mode='r')
tnn_ee = era_tnn_e.variables['tnn'][:,:,:]





def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


rmse_tnn_d = np.round(rmse(tnn_ee, tnn_ae), decimals=3)

sc_nd, pv = stats.spearmanr(np.ravel(tnn_ee), np.ravel(tnn_ae), axis=0)
sc_nd = np.round(sc_nd, decimals=3)

#SON
awap_tnn_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ap_3m_SON.nc', mode='r')
tnn_al = awap_tnn_l.variables['tnn'][:,:,:]


era_tnn_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/tnn_ei_3m_SON.nc', mode='r')
tnn_el = era_tnn_l.variables['tnn'][:,:,:]


rmse_tnn_s = np.round(rmse(tnn_el, tnn_al), decimals=3)


sc_ns, pv = stats.spearmanr(np.ravel(tnn_el), np.ravel(tnn_al), axis=0)
sc_ns = np.round(sc_ns, decimals=3)



#DJF
#TXx
awap_txx_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ap_3m_DJF.nc', mode='r')

lon = awap_txx_e.variables['lon'][:]
lat = awap_txx_e.variables['lat'][:]
txx_ae = awap_txx_e.variables['txx'][:,:,:]
units = awap_txx_e.variables['time'].units
time_nino = awap_txx_e.variables['time'][:]

era_txx_e = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ei_3m_DJF.nc', mode='r')
txx_ee = era_txx_e.variables['txx'][:,:,:]



rmse_txx_d = np.round(rmse(txx_ee, txx_ae), decimals=3)

sc_xd, pv = stats.spearmanr(np.ravel(txx_ee), np.ravel(txx_ae), axis=0)
sc_xd = np.round(sc_xd, decimals=3)

#SON
awap_txx_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ap_3m_SON.nc', mode='r')
txx_al = awap_txx_l.variables['txx'][:,:,:]


era_txx_l = Dataset('/srv/ccrc/data06/z5147939/ncfiles/ds_comp_hr/txx_ei_3m_SON.nc', mode='r')
txx_el = era_txx_l.variables['txx'][:,:,:]

rmse_txx_s = np.round(rmse(txx_el, txx_al), decimals=4)
rmse_txx_s = "%.3f" % rmse_txx_s
sc_xs, pv = stats.spearmanr(np.ravel(txx_el), np.ravel(txx_al), axis=0)
sc_xs = np.round(sc_xs, decimals=3)




#define empty arrays
corr_n_d = np.zeros((len(lat), len(lon)))
pv_n_d = np.zeros((len(lat), len(lon)))
ss_n_d = np.zeros((len(lat), len(lon)))

corr_n_s = np.zeros((len(lat), len(lon)))
pv_n_s = np.zeros((len(lat), len(lon)))
ss_n_s = np.zeros((len(lat), len(lon)))

corr_x_d = np.zeros((len(lat), len(lon)))
pv_x_d = np.zeros((len(lat), len(lon)))
ss_x_d = np.zeros((len(lat), len(lon)))

corr_x_s = np.zeros((len(lat), len(lon)))
pv_x_s = np.zeros((len(lat), len(lon)))
ss_x_s = np.zeros((len(lat), len(lon)))

#loop correlations
for i in range(0,len(lon)):
    for j in range(0,len(lat)):
      corr_n_d[j,i], pv_n_d[j,i] =  stats.spearmanr(tnn_ee[:,j,i], tnn_ae[:,j,i], axis=0)
      if pv_n_d[j,i] < 0.05:
        ss_n_d[j,i] = corr_n_d[j,i]
      else:
        ss_n_d[j,i] = np.NaN
      corr_n_s[j,i], pv_n_s[j,i] =  stats.spearmanr(tnn_el[:,j,i], tnn_al[:,j,i], axis=0)
      if pv_n_s[j,i] < 0.05:
        ss_n_s[j,i] = corr_n_s[j,i]
      else:
        ss_n_s[j,i] = np.NaN
      corr_x_d[j,i], pv_x_d[j,i] =  stats.spearmanr(txx_ee[:,j,i], txx_ae[:,j,i], axis=0)
      if pv_x_d[j,i] < 0.05:
        ss_x_d[j,i] = corr_x_d[j,i]
      else:
        ss_x_d[j,i] = np.NaN
      corr_x_s[j,i], pv_x_s[j,i] =  stats.spearmanr(txx_el[:,j,i], txx_al[:,j,i], axis=0)
      if pv_x_s[j,i] < 0.05:
        ss_x_s[j,i] = corr_x_s[j,i]
      else:
        ss_x_s[j,i] = np.NaN

#corr_n_d = np.ma.masked_invalid(corr_n_d)
#corr_n_s = np.ma.masked_invalid(corr_n_s)
#corr_x_d = np.ma.masked_invalid(corr_x_d)
#corr_x_s = np.ma.masked_invalid(corr_x_s)


#print np.ma.max(corr_n_d), np.ma.min(corr_n_d)
#print np.ma.max(corr_n_s), np.ma.min(corr_n_s)
#print np.ma.max(corr_x_d), np.ma.min(corr_x_d)
#print np.ma.max(corr_x_s), np.ma.min(corr_x_s)
#print stop


#plot
#define global attributes
lons, lats = np.meshgrid(lon,lat)
v = np.linspace( 0.2, 1.0, 17, endpoint=True)
norm = colors.BoundaryNorm(boundaries=v, ncolors=256)
cmap = plt.cm.jet


from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

orig_cmap = matplotlib.cm.bwr
cmap_shifted = shiftedColorMap(orig_cmap, midpoint=0.15, name='shifted')

#plot TXx during El Nino
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(222)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0) #define basemap as around Australia
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
xi,yi = m(lons,lats)
corr_n_d = np.ma.masked_invalid(corr_n_d)
ss_n_d = np.ma.masked_invalid(ss_n_d)
mymap= m.pcolor(xi, yi, corr_n_d, norm=norm, cmap=cmap)
mymap= m.pcolor(xi, yi, ss_n_d, hatch='...', norm=norm, cmap=cmap)
plt.title('DJF TNn', fontsize=12)


#plot TNn during El Nino
ax = plt.subplot(221)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
corr_n_s = np.ma.masked_invalid(corr_n_s)
ss_n_s = np.ma.masked_invalid(ss_n_s)
mymap= m.pcolor(xi, yi, corr_n_s, norm=norm, cmap=cmap)
mymap= m.pcolor(xi, yi, ss_n_s, hatch='...', norm=norm, cmap=cmap)
plt.title('SON TNn', fontsize=12)
#ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_tnn_s, sc_ns), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')


#plot txx during la nina
ax = plt.subplot(224)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
corr_x_d = np.ma.masked_invalid(corr_x_d)
ss_x_d = np.ma.masked_invalid(ss_x_d)
mymap= m.pcolor(xi, yi, corr_x_d, norm=norm, cmap=cmap)
mymap= m.pcolor(xi, yi, ss_x_d, hatch='...', norm=norm, cmap=cmap)
plt.title('DJF TXx', fontsize=12)
#ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_txx_d, sc_xd), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')


#plot tnn during la nina
ax = plt.subplot(223)
m = Basemap(projection='cyl', llcrnrlat=-45.0, llcrnrlon=110.0, urcrnrlat=-5.0, urcrnrlon=160.0)
m.drawcoastlines()
m.drawparallels(np.array([-45, -35, -25, -15, -5]), labels=[1,0,0,0], fontsize=8)
m.drawmeridians(np.array([110, 120, 130, 140, 150, 160]), labels=[0,0,0,1], fontsize=8)
corr_x_s = np.ma.masked_invalid(corr_x_s)
ss_x_s = np.ma.masked_invalid(ss_x_s)
mymap= m.pcolor(xi, yi, corr_x_s, norm=norm, cmap=cmap)
mymap= m.pcolor(xi, yi, ss_x_s, hatch='...', norm=norm, cmap=cmap)
plt.title('SON TXx', fontsize=12)
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
cb = fig.colorbar(mymap, cax, orientation='vertical')
cb.ax.tick_params(labelsize=8)
# for label in cb.ax.yaxis.get_ticklabels()[1::2]:
#     label.set_visible(False)
cb.set_label('Correlation', fontsize=8)
#ax.text(0.99,0.01,'RMSE = %s, CORR = %s' % (rmse_txx_s, sc_xs), transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=8, fontweight='bold')
plt.savefig('/home/z5147939/hdrive/figs/era_corr.png', bbox_inches='tight')
#plt.show()


#print np.max(d_txx_e), np.min(d_txx_e)
#print np.max(d_txx_l), np.min(d_txx_l)
#print np.max(d_tnn_e), np.min(d_tnn_e)
#print np.max(d_tnn_l), np.min(d_tnn_l)
