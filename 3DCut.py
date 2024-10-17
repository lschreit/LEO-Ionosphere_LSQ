#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:30:41 2023

Script to generate a 3D cut of the electron density in the ionosphere.
Possible identifiers for tst are:
    "_PaperFin_002_PWU"
    "_PaperFin_002_NoSpire_PWU"
    
For the first option the boolean filt can be set to True to display electron 
density obtained from smoothed model coefficients.

@author: schreit
"""

import numpy as np
import matplotlib.pyplot as plt
from apexpy import apex
import matplotlib as mpl
from datetime import datetime


from NeModel import NeModel as NeM

import pylab
from datetime import timedelta

filt=False

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


Apx=apex.Apex(2020)
tst_rmse=[]
tst_cc=[]
day_start=datetime(2020,5,3,3)
day_end=day_start+timedelta(days=1)
tst="_PaperFin_002_PWU"

year=day_start.year
day=day_start
doy=day_start.timetuple().tm_yday

dirDay = "MOD/"+str(year)+str(doy).zfill(3)+tst

Nem  = NeM(dirDay,filt=filt)

MLATs    = np.linspace(-90,90,91)
MLTs     = np.linspace(0,24,91)[:-1]
Heights  = np.array([120,400,850])

mlatv,mltv,heightv = np.meshgrid(MLATs,MLTs,Heights)
shp=mlatv.shape
mlat   = mlatv.flatten()
mlt    = mltv.flatten()
height = heightv.flatten()
ts=[day for _ in mlat]
    

res   = Nem.evaluate_model(mlat, mlt, height, dati=ts, coord="mlt")
res=res.reshape(shp)
res=np.log10(res)

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=2, vmax=6.2)



for i in range(mlatv.shape[2]):
    p=ax.contourf(mltv[:,:,i],mlatv[:,:,i],Heights[i]+0.01*res[:,:,i],zdir='z',levels=Heights[i]+0.01*np.linspace(2,6.2,22),cmap=cmap)
    
    # ax.contourf(X, Y, .1*np.sin(3*X)*np.sin(5*Y), zdir='z', levels=.1*levels)
    





ax.set_xlim3d(0, 24)
ax.set_ylim3d(-90, 90)
ax.set_zlim3d(100, 850)

plt.title(str(ts[0])+" UTC")
ax.set_zlabel("Altitude [km]")
ax.set_ylabel("Magnetic latitude [deg.]")
ax.set_xlabel("Magnetic local time [h]")
tick=0.01*np.float64([2, 3,4,5,6])+ Heights[-1]
cbar=fig.colorbar(p,label="$log_{10} (Ne)$",location="left")
cbar.set_ticks(tick)
cbar.set_ticklabels(["2","3","4","5","6"])
plt.show()
pylab.savefig("3DHori.png",dpi=400,bbox_inches=None)
