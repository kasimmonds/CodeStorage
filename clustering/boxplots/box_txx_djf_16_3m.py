# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats



nc2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_2_sil_0.1.nc', mode='r')
sil2 = nc2.variables['sil_width'][:,:]
sil2 = np.ma.compressed(sil2)

nc3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_3_sil_0.1.nc', mode='r')
sil3 = nc3.variables['sil_width'][:,:]
sil3 = np.ma.compressed(sil3)

nc4 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_4_sil_0.1.nc', mode='r')
sil4 = nc4.variables['sil_width'][:,:]
sil4 = np.ma.compressed(sil4)

nc5 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_5_sil_0.1.nc', mode='r')
sil5 = nc5.variables['sil_width'][:,:]
sil5 = np.ma.compressed(sil5)

nc6 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_6_sil_0.1.nc', mode='r')
sil6 = nc6.variables['sil_width'][:,:]
sil6 = np.ma.compressed(sil6)

nc7 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_7_sil_0.1.nc', mode='r')
sil7 = nc7.variables['sil_width'][:,:]
sil7 = np.ma.compressed(sil7)

nc8 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_8_sil_0.1.nc', mode='r')
sil8 = nc8.variables['sil_width'][:,:]
sil8 = np.ma.compressed(sil8)

nc9 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_9_sil_0.1.nc', mode='r')
sil9 = nc9.variables['sil_width'][:,:]
sil9 = np.ma.compressed(sil9)

nc10 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_10_sil_0.1.nc', mode='r')
sil10 = nc10.variables['sil_width'][:,:]
sil10 = np.ma.compressed(sil10)

nc11 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_11_sil_0.1.nc', mode='r')
sil11 = nc11.variables['sil_width'][:,:]
sil11 = np.ma.compressed(sil11)

nc12 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_12_sil_0.1.nc', mode='r')
sil12 = nc12.variables['sil_width'][:,:]
sil12 = np.ma.compressed(sil12)

nc13 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_13_sil_0.1.nc', mode='r')
sil13 = nc13.variables['sil_width'][:,:]
sil13 = np.ma.compressed(sil13)

nc14 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_14_sil_0.1.nc', mode='r')
sil14 = nc14.variables['sil_width'][:,:]
sil14 = np.ma.compressed(sil14)

nc15 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_15_sil_0.1.nc', mode='r')
sil15 = nc15.variables['sil_width'][:,:]
sil15 = np.ma.compressed(sil15)

nc16 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_16_sil_0.1.nc', mode='r')
sil16 = nc16.variables['sil_width'][:,:]
sil16 = np.ma.compressed(sil16)

nc17 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_17_sil_0.1.nc', mode='r')
sil17 = nc17.variables['sil_width'][:,:]
sil17 = np.ma.compressed(sil17)

nc18 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_18_sil_0.1.nc', mode='r')
sil18 = nc18.variables['sil_width'][:,:]
sil18 = np.ma.compressed(sil18)

nc19 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_19_sil_0.1.nc', mode='r')
sil19 = nc19.variables['sil_width'][:,:]
sil19 = np.ma.compressed(sil19)

nc20 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_20_sil_0.1.nc', mode='r')
sil20 = nc20.variables['sil_width'][:,:]
sil20 = np.ma.compressed(sil20)

sil_box = [sil2, sil3, sil4, sil5, sil6, sil7, sil8, sil9, sil10, sil11, sil12, sil13, sil14, sil15, sil16, sil17, sil18, sil19, sil20]

#set up figure
plt.figure(1, figsize=(11, 4))
ax = plt.subplot(111)

#plot boxplot with brackets at 10th and 90th percentile
ax.boxplot(sil_box, whis=[10,90],showfliers=False)
ax.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
ax.set_xlabel('Number of Clusters', fontsize=10)
ax.set_ylabel('Silhouette Coefficient', fontsize=10)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_title('TXx DJF 1979-2016', fontsize=12)
plt.ylim(-0.1,0.4)
plt.axhline(y=0.1, color='k', linestyle='dashed')
ax.tick_params(axis='both', which='major', labelsize=8)
plt.savefig('/home/z5147939/hdrive/figs/txx_box_djf_16_3m.png', bbox_inches='tight')
plt.show()
