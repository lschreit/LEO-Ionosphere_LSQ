#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:29:35 2023

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
# Heights  = [120,250,250,500,850]
Heights  = np.linspace(100,1000,100)

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

k=1/(1e13)

lvl=+k*np.linspace(2,6.2,22)
# lvl=k*np.linspace(0,1e6,22)


for i in range(mlatv.shape[0]):
    if (i%30)==0:
        p=ax.contourf(MLTs[i]+k*res[i,:,:],mlatv[i,:,:],heightv[i,:,:],zdir='x',levels=MLTs[i]+lvl,cmap=cmap)
        bla=MLTs[i]


ax.set_xlim3d(0, 24)
ax.set_ylim3d(-90, 90)
ax.set_zlim3d(100, 850)

plt.title(str(ts[0])+" UTC")
ax.set_zlabel("Altitude [km]")
ax.set_ylabel("Magnetic latitude [deg.]")
ax.set_xlabel("Magnetic local time [h]")
tick=k*np.float64([2, 3,4,5,6])+ bla
cbar=fig.colorbar(p,label="$log_{10} (Ne)$",location="left")


cbar.set_ticks(tick)
cbar.set_ticklabels(["2","3","4","5","6"])

plt.show()
pylab.savefig("3DVert_LSQ.png",dpi=400,bbox_inches=None)
