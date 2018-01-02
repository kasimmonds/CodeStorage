# -*- coding: utf-8 -*-


#import neccessary modules
from netCDF4 import Dataset
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy import stats



nc2 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_2_sil_0.1.nc', mode='r')
sil2 = nc2.variables['sil_width'][:,:]
sil2 = np.ma.compressed(sil2)

nc3 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_3_sil_0.1.nc', mode='r')
sil3 = nc3.variables['sil_width'][:,:]
sil3 = np.ma.compressed(sil3)

nc4 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_4_sil_0.1.nc', mode='r')
sil4 = nc4.variables['sil_width'][:,:]
sil4 = np.ma.compressed(sil4)

nc5 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_5_sil_0.1.nc', mode='r')
sil5 = nc5.variables['sil_width'][:,:]
sil5 = np.ma.compressed(sil5)

nc6 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_6_sil_0.1.nc', mode='r')
sil6 = nc6.variables['sil_width'][:,:]
sil6 = np.ma.compressed(sil6)

nc7 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_7_sil_0.1.nc', mode='r')
sil7 = nc7.variables['sil_width'][:,:]
sil7 = np.ma.compressed(sil7)

nc8 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_8_sil_0.1.nc', mode='r')
sil8 = nc8.variables['sil_width'][:,:]
sil8 = np.ma.compressed(sil8)

nc9 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_9_sil_0.1.nc', mode='r')
sil9 = nc9.variables['sil_width'][:,:]
sil9 = np.ma.compressed(sil9)

nc10 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_10_sil_0.1.nc', mode='r')
sil10 = nc10.variables['sil_width'][:,:]
sil10 = np.ma.compressed(sil10)

nc11 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_11_sil_0.1.nc', mode='r')
sil11 = nc11.variables['sil_width'][:,:]
sil11 = np.ma.compressed(sil11)

nc12 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_12_sil_0.1.nc', mode='r')
sil12 = nc12.variables['sil_width'][:,:]
sil12 = np.ma.compressed(sil12)

nc13 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_13_sil_0.1.nc', mode='r')
sil13 = nc13.variables['sil_width'][:,:]
sil13 = np.ma.compressed(sil13)

nc14 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_14_sil_0.1.nc', mode='r')
sil14 = nc14.variables['sil_width'][:,:]
sil14 = np.ma.compressed(sil14)

nc15 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_15_sil_0.1.nc', mode='r')
sil15 = nc15.variables['sil_width'][:,:]
sil15 = np.ma.compressed(sil15)

nc16 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_16_sil_0.1.nc', mode='r')
sil16 = nc16.variables['sil_width'][:,:]
sil16 = np.ma.compressed(sil16)

nc17 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_17_sil_0.1.nc', mode='r')
sil17 = nc17.variables['sil_width'][:,:]
sil17 = np.ma.compressed(sil17)

nc18 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_18_sil_0.1.nc', mode='r')
sil18 = nc18.variables['sil_width'][:,:]
sil18 = np.ma.compressed(sil18)

nc19 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_19_sil_0.1.nc', mode='r')
sil19 = nc19.variables['sil_width'][:,:]
sil19 = np.ma.compressed(sil19)

nc20 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_20_sil_0.1.nc', mode='r')
sil20 = nc20.variables['sil_width'][:,:]
sil20 = np.ma.compressed(sil20)

nc21 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_21_sil_0.1.nc', mode='r')
sil21 = nc11.variables['sil_width'][:,:]
sil21 = np.ma.compressed(sil21)

nc22 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_22_sil_0.1.nc', mode='r')
sil22 = nc12.variables['sil_width'][:,:]
sil22 = np.ma.compressed(sil22)

nc23 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_23_sil_0.1.nc', mode='r')
sil23 = nc13.variables['sil_width'][:,:]
sil23 = np.ma.compressed(sil23)

nc24 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_24_sil_0.1.nc', mode='r')
sil24 = nc14.variables['sil_width'][:,:]
sil24 = np.ma.compressed(sil24)

nc25 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_25_sil_0.1.nc', mode='r')
sil25 = nc15.variables['sil_width'][:,:]
sil25 = np.ma.compressed(sil25)

nc26 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_26_sil_0.1.nc', mode='r')
sil26 = nc16.variables['sil_width'][:,:]
sil26 = np.ma.compressed(sil26)

nc27 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_27_sil_0.1.nc', mode='r')
sil27 = nc17.variables['sil_width'][:,:]
sil27 = np.ma.compressed(sil27)

nc28 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_28_sil_0.1.nc', mode='r')
sil28 = nc18.variables['sil_width'][:,:]
sil28 = np.ma.compressed(sil28)

nc29 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_29_sil_0.1.nc', mode='r')
sil29 = nc19.variables['sil_width'][:,:]
sil29 = np.ma.compressed(sil29)

nc30 = Dataset('/home/z5147939/ncfiles/clust_era/txx_SON_K_30_sil_0.1.nc', mode='r')
sil30 = nc20.variables['sil_width'][:,:]
sil30 = np.ma.compressed(sil30)

#define each silhouette coefficient as a list
sil_box = [sil2, sil3, sil4, sil5, sil6, sil7, sil8, sil9, sil10, sil11, sil12, sil13, sil14, sil15, sil16, sil17, sil18, sil19, sil20, sil21, sil22, sil23, sil24, sil25, sil26, sil27, sil28, sil29, sil30]



anc2 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_2_sil_0.1.nc', mode='r')
asil2 = anc2.variables['sil_width'][:,:]
asil2 = np.ma.compressed(asil2)

anc3 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_3_sil_0.1.nc', mode='r')
asil3 = anc3.variables['sil_width'][:,:]
asil3 = np.ma.compressed(asil3)

anc4 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_4_sil_0.1.nc', mode='r')
asil4 = anc4.variables['sil_width'][:,:]
asil4 = np.ma.compressed(asil4)

anc5 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_5_sil_0.1.nc', mode='r')
asil5 = anc5.variables['sil_width'][:,:]
asil5 = np.ma.compressed(asil5)

anc6 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_6_sil_0.1.nc', mode='r')
asil6 = anc6.variables['sil_width'][:,:]
asil6 = np.ma.compressed(asil6)

anc7 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_7_sil_0.1.nc', mode='r')
asil7 = anc7.variables['sil_width'][:,:]
asil7 = np.ma.compressed(asil7)

anc8 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_8_sil_0.1.nc', mode='r')
asil8 = nc8.variables['sil_width'][:,:]
asil8 = np.ma.compressed(sil8)

anc9 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_9_sil_0.1.nc', mode='r')
asil9 = anc9.variables['sil_width'][:,:]
asil9 = np.ma.compressed(asil9)

anc10 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_10_sil_0.1.nc', mode='r')
asil10 = anc10.variables['sil_width'][:,:]
asil10 = np.ma.compressed(asil10)

anc11 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_11_sil_0.1.nc', mode='r')
asil11 = anc11.variables['sil_width'][:,:]
asil11 = np.ma.compressed(asil11)

anc12 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_12_sil_0.1.nc', mode='r')
asil12 = anc12.variables['sil_width'][:,:]
asil12 = np.ma.compressed(asil12)

anc13 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_13_sil_0.1.nc', mode='r')
asil13 = anc13.variables['sil_width'][:,:]
asil13 = np.ma.compressed(asil13)

anc14 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_14_sil_0.1.nc', mode='r')
asil14 = anc14.variables['sil_width'][:,:]
asil14 = np.ma.compressed(asil14)

anc15 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_15_sil_0.1.nc', mode='r')
asil15 = anc15.variables['sil_width'][:,:]
asil15 = np.ma.compressed(asil15)

anc16 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_16_sil_0.1.nc', mode='r')
asil16 = anc16.variables['sil_width'][:,:]
asil16 = np.ma.compressed(asil16)

anc17 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_17_sil_0.1.nc', mode='r')
asil17 = anc17.variables['sil_width'][:,:]
asil17 = np.ma.compressed(asil17)

anc18 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_18_sil_0.1.nc', mode='r')
asil18 = anc18.variables['sil_width'][:,:]
asil18 = np.ma.compressed(asil18)

anc19 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_19_sil_0.1.nc', mode='r')
asil19 = nc19.variables['sil_width'][:,:]
asil19 = np.ma.compressed(asil19)

anc20 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_20_sil_0.1.nc', mode='r')
asil20 = anc20.variables['sil_width'][:,:]
asil20 = np.ma.compressed(asil20)

anc21 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_21_sil_0.1.nc', mode='r')
asil21 = anc11.variables['sil_width'][:,:]
asil21 = np.ma.compressed(asil21)

anc22 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_22_sil_0.1.nc', mode='r')
asil22 = anc12.variables['sil_width'][:,:]
asil22 = np.ma.compressed(asil22)

anc23 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_23_sil_0.1.nc', mode='r')
asil23 = anc13.variables['sil_width'][:,:]
asil23 = np.ma.compressed(asil23)

anc24 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_24_sil_0.1.nc', mode='r')
asil24 = anc14.variables['sil_width'][:,:]
asil24 = np.ma.compressed(asil24)

anc25 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_25_sil_0.1.nc', mode='r')
asil25 = anc15.variables['sil_width'][:,:]
asil25 = np.ma.compressed(asil25)

anc26 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_26_sil_0.1.nc', mode='r')
asil26 = anc16.variables['sil_width'][:,:]
asil26 = np.ma.compressed(asil26)

anc27 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_27_sil_0.1.nc', mode='r')
asil27 = anc17.variables['sil_width'][:,:]
asil27 = np.ma.compressed(asil27)

anc28 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_28_sil_0.1.nc', mode='r')
asil28 = anc18.variables['sil_width'][:,:]
asil28 = np.ma.compressed(asil28)

anc29 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_29_sil_0.1.nc', mode='r')
asil29 = anc19.variables['sil_width'][:,:]
asil29 = np.ma.compressed(asil29)

anc30 = Dataset('/home/z5147939/ncfiles/clust_era/txx_ap_SON_K_30_sil_0.1.nc', mode='r')
asil30 = anc20.variables['sil_width'][:,:]
asil30 = np.ma.compressed(asil30)

#define each silhouette coefficient as a list
asil_box = [asil2, asil3, asil4, asil5, asil6, asil7, asil8, asil9, asil10, asil11, asil12, asil13, asil14, asil15, asil16, asil17, asil18, asil19, asil20, asil21, asil22, asil23, asil24, asil25, asil26, asil27, asil28, asil29, asil30]

#set up figure
fig = plt.figure(1, figsize=(12, 10))
ax = fig.add_subplot(211)

#plot boxplot with brackets at 10th and 90th percentile
bp  = ax.boxplot(sil_box, whis=[10,90],showfliers=False)
ax.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
#ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Coefficient')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_title('ERA-Interim TXx SON 1979-2013')
plt.ylim(-0.1,0.4)
plt.axhline(y=0.1, color='k', linestyle='dashed')

ax = fig.add_subplot(212)
bp  = ax.boxplot(asil_box, whis=[10,90],showfliers=False)
ax.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Coefficient')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_title('AWAP TXx SON 1979-2013')
plt.ylim(-0.1,0.4)
plt.axhline(y=0.1, color='k', linestyle='dashed')
plt.savefig('/home/z5147939/hdrive/figs/txx_box_son.png', bbox_inches='tight')
plt.show()
