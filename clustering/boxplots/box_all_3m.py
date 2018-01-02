# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats



nc2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_2_sil_0.1.nc', mode='r')
nc2_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_2_sil_0.1.nc', mode='r')
nc2_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_2_sil_0.1.nc', mode='r')
nc2_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_2_sil_0.1.nc', mode='r')
sil2 = np.ma.compressed((nc2.variables['sil_width'][:,:] + nc2_1.variables['sil_width'][:,:] + nc2_2.variables['sil_width'][:,:] + nc2_3.variables['sil_width'][:,:])/4)


nc3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_3_sil_0.1.nc', mode='r')
nc3_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_3_sil_0.1.nc', mode='r')
nc3_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_3_sil_0.1.nc', mode='r')
nc3_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_3_sil_0.1.nc', mode='r')
sil3 = np.ma.compressed((nc3.variables['sil_width'][:,:] + nc3_1.variables['sil_width'][:,:] + nc3_2.variables['sil_width'][:,:] + nc3_3.variables['sil_width'][:,:])/4)

nc4 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_4_sil_0.1.nc', mode='r')
nc4_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_4_sil_0.1.nc', mode='r')
nc4_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_4_sil_0.1.nc', mode='r')
nc4_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_4_sil_0.1.nc', mode='r')
sil4 = np.ma.compressed((nc4.variables['sil_width'][:,:] + nc4_1.variables['sil_width'][:,:] + nc4_2.variables['sil_width'][:,:] + nc4_3.variables['sil_width'][:,:])/4)

nc5 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_5_sil_0.1.nc', mode='r')
nc5_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_5_sil_0.1.nc', mode='r')
nc5_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_5_sil_0.1.nc', mode='r')
nc5_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_5_sil_0.1.nc', mode='r')
sil5 = np.ma.compressed((nc5.variables['sil_width'][:,:] + nc5_1.variables['sil_width'][:,:] + nc5_2.variables['sil_width'][:,:] + nc4_3.variables['sil_width'][:,:])/4)

nc6 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_6_sil_0.1.nc', mode='r')
nc6_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_6_sil_0.1.nc', mode='r')
nc6_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_6_sil_0.1.nc', mode='r')
nc6_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_6_sil_0.1.nc', mode='r')
sil6 = np.ma.compressed((nc6.variables['sil_width'][:,:] + nc6_1.variables['sil_width'][:,:] + nc6_2.variables['sil_width'][:,:] + nc6_3.variables['sil_width'][:,:])/4)


nc7 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_7_sil_0.1.nc', mode='r')
nc7_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_7_sil_0.1.nc', mode='r')
nc7_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_7_sil_0.1.nc', mode='r')
nc7_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_7_sil_0.1.nc', mode='r')
sil7 = np.ma.compressed((nc7.variables['sil_width'][:,:] + nc7_1.variables['sil_width'][:,:] + nc7_2.variables['sil_width'][:,:] + nc7_3.variables['sil_width'][:,:])/4)

nc8 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_8_sil_0.1.nc', mode='r')
nc8_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_8_sil_0.1.nc', mode='r')
nc8_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_8_sil_0.1.nc', mode='r')
nc8_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_8_sil_0.1.nc', mode='r')
sil8 = np.ma.compressed((nc8.variables['sil_width'][:,:] + nc8_1.variables['sil_width'][:,:] + nc8_2.variables['sil_width'][:,:] + nc8_3.variables['sil_width'][:,:])/4)

nc9 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_9_sil_0.1.nc', mode='r')
nc9_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_9_sil_0.1.nc', mode='r')
nc9_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_9_sil_0.1.nc', mode='r')
nc9_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_9_sil_0.1.nc', mode='r')
sil9 = np.ma.compressed((nc9.variables['sil_width'][:,:] + nc9_1.variables['sil_width'][:,:] + nc9_2.variables['sil_width'][:,:] + nc9_3.variables['sil_width'][:,:])/4)

nc10 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_10_sil_0.1.nc', mode='r')
nc10_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_10_sil_0.1.nc', mode='r')
nc10_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_10_sil_0.1.nc', mode='r')
nc10_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_10_sil_0.1.nc', mode='r')
sil10 = np.ma.compressed((nc10.variables['sil_width'][:,:] + nc10_1.variables['sil_width'][:,:] + nc10_2.variables['sil_width'][:,:] + nc10_3.variables['sil_width'][:,:])/4)

nc11 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_11_sil_0.1.nc', mode='r')
nc11_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_11_sil_0.1.nc', mode='r')
nc11_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_11_sil_0.1.nc', mode='r')
nc11_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_11_sil_0.1.nc', mode='r')
sil11 = np.ma.compressed((nc11.variables['sil_width'][:,:] + nc11_1.variables['sil_width'][:,:] + nc11_2.variables['sil_width'][:,:] + nc11_3.variables['sil_width'][:,:])/4)

nc12 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_12_sil_0.1.nc', mode='r')
nc12_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_12_sil_0.1.nc', mode='r')
nc12_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_12_sil_0.1.nc', mode='r')
nc12_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_12_sil_0.1.nc', mode='r')
sil12 = np.ma.compressed((nc12.variables['sil_width'][:,:] + nc12_1.variables['sil_width'][:,:] + nc12_2.variables['sil_width'][:,:] + nc12_3.variables['sil_width'][:,:])/4)

nc13 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_13_sil_0.1.nc', mode='r')
nc13_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_13_sil_0.1.nc', mode='r')
nc13_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_13_sil_0.1.nc', mode='r')
nc13_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_13_sil_0.1.nc', mode='r')
sil13 = np.ma.compressed((nc13.variables['sil_width'][:,:] + nc13_1.variables['sil_width'][:,:] + nc13_3.variables['sil_width'][:,:] + nc13_3.variables['sil_width'][:,:])/4)

nc14 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_14_sil_0.1.nc', mode='r')
nc14_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_14_sil_0.1.nc', mode='r')
nc14_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_14_sil_0.1.nc', mode='r')
nc14_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_14_sil_0.1.nc', mode='r')
sil14 = np.ma.compressed((nc14.variables['sil_width'][:,:] + nc14_1.variables['sil_width'][:,:] + nc14_3.variables['sil_width'][:,:] + nc14_3.variables['sil_width'][:,:])/4)

nc15 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_15_sil_0.1.nc', mode='r')
nc15_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_15_sil_0.1.nc', mode='r')
nc15_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_15_sil_0.1.nc', mode='r')
nc15_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_15_sil_0.1.nc', mode='r')
sil15 = np.ma.compressed((nc15.variables['sil_width'][:,:] + nc15_1.variables['sil_width'][:,:] + nc15_3.variables['sil_width'][:,:] + nc15_3.variables['sil_width'][:,:])/4)

nc16 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_16_sil_0.1.nc', mode='r')
nc16_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_16_sil_0.1.nc', mode='r')
nc16_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_16_sil_0.1.nc', mode='r')
nc16_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_16_sil_0.1.nc', mode='r')
sil16 = np.ma.compressed((nc16.variables['sil_width'][:,:] + nc16_1.variables['sil_width'][:,:] + nc16_3.variables['sil_width'][:,:] + nc16_3.variables['sil_width'][:,:])/4)

nc17 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_17_sil_0.1.nc', mode='r')
nc17_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_17_sil_0.1.nc', mode='r')
nc17_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_17_sil_0.1.nc', mode='r')
nc17_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_17_sil_0.1.nc', mode='r')
sil17 = np.ma.compressed((nc17.variables['sil_width'][:,:] + nc17_1.variables['sil_width'][:,:] + nc17_3.variables['sil_width'][:,:] + nc17_3.variables['sil_width'][:,:])/4)

nc18 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_18_sil_0.1.nc', mode='r')
nc18_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_18_sil_0.1.nc', mode='r')
nc18_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_18_sil_0.1.nc', mode='r')
nc18_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_18_sil_0.1.nc', mode='r')
sil18 = np.ma.compressed((nc18.variables['sil_width'][:,:] + nc18_1.variables['sil_width'][:,:] + nc18_3.variables['sil_width'][:,:] + nc18_3.variables['sil_width'][:,:])/4)

nc19 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_19_sil_0.1.nc', mode='r')
nc19_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_19_sil_0.1.nc', mode='r')
nc19_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_19_sil_0.1.nc', mode='r')
nc19_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_19_sil_0.1.nc', mode='r')
sil19 = np.ma.compressed((nc19.variables['sil_width'][:,:] + nc19_1.variables['sil_width'][:,:] + nc19_3.variables['sil_width'][:,:] + nc19_3.variables['sil_width'][:,:])/4)

nc20 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_DJF_2016_K_20_sil_0.1.nc', mode='r')
nc20_1 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_DJF_2016_K_20_sil_0.1.nc', mode='r')
nc20_2 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/tnn_SON_2016_K_20_sil_0.1.nc', mode='r')
nc20_3 = Dataset('/srv/ccrc/data06/z5147939/ncfiles/clust_3m/txx_SON_2016_K_20_sil_0.1.nc', mode='r')
sil20 = np.ma.compressed((nc20.variables['sil_width'][:,:] + nc20_1.variables['sil_width'][:,:] + nc20_3.variables['sil_width'][:,:] + nc20_3.variables['sil_width'][:,:])/4)

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
ax.set_title('Mean silhouette coefficient statistics for TNn/TXx in SON/DJF', fontsize=12)
plt.ylim(0,0.4)
plt.axhline(y=0.1, color='k', linestyle='dashed')
ax.tick_params(axis='both', which='major', labelsize=8)
plt.savefig('/home/z5147939/hdrive/figs/box_mean_3m.png', bbox_inches='tight')
plt.show()
