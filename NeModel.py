#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:51:54 2023
Evaluate model
@author: schreit
"""
import numpy as np
from scipy import sparse
import apexpy
from SplineBase import SplineBase

class NeModel:
    
    def __init__(self,dirDay,filt=False):
        """
        Knot Vectors are assumed to be in 00 Folder

        Parameters
        ----------
        dirDay : str
            Typically something like "MOD/"+str(year)+doy+ident.

        Returns
        -------
        None.

        """
        self.rng=24
        hour="00"
        if not filt:
            self.Mod_par = np.array([np.load(
                "./"+dirDay+"/"+str(dirHour).zfill(2)+"/DATA/ModPar.npy") for dirHour in range(self.rng)])
        if filt:
            self.Mod_par = np.array([np.load(
                "./"+dirDay+"/"+str(dirHour).zfill(2)+"/DATA/ModPar_filt.npy") for dirHour in range(self.rng)])
            
        KnotmLat = np.load("./"+dirDay+"/"+hour+"/DATA/Knot_MLAT.npy")
        KnotLT = np.load("./"+dirDay+"/"+hour+"/DATA/Knot_MLT.npy")
        KnotH = np.load("./"+dirDay+"/"+hour+"/DATA/Knot_H.npy")
        self.SB = SplineBase(KnotmLat, KnotLT, KnotH)
        # self.nlat = len(KnotmLat)
        # self.nlt = len(KnotLT)
        # self.nh = len(KnotH)
        # self.npar = nlt*(nlat+2)*(nh+2)
        
    def compute_model(self,mlat : np.array,mlt: np.array,heights: np.array,hourFloat: np.array):
        """
        

        Parameters
        ----------
        mlat : np.array
            magnetic latitudes as array.
        mlt : np.array
            magnetic local times as array.
        heights : np.array
            heighs as array.
        hourFloat : np.array
            timestamp as floating point hour.

        Returns
        -------
        neo : np.array
            electron density values.

        """
        hourFloat += 0.5 
        mlat_f = np.array(mlat)
        mlt_f = np.array(mlt)
        neo = np.zeros(len(mlat_f))+np.nan

        ne1 = np.zeros(len(mlat))
        ne2 = np.zeros(len(mlat))
    
        w = hourFloat-np.int8(hourFloat)
        print(mlat_f)
        print(mlt_f)
        print(heights)
        mat_full = self.SB.get_spline_base_full(mlat_f, mlt_f, heights)
        mat_full = sparse.csr_matrix(mat_full)
        tst = np.maximum(np.minimum(24, np.int8(hourFloat)), 0)
       

        for hhh in range(self.rng+1):
            index = np.where(hhh == tst)

            hh0 = min(max(0, hhh-1), self.rng-1)
            hh1 = min(self.rng-1, hhh)

            mat = mat_full[index]
            ne1[index] = mat.dot(self.Mod_par[hh0])
            ne2[index] = mat.dot(self.Mod_par[hh1])

        neo = (1-w)*ne1+w*ne2

        return neo
    
    def evaluate_model(self, lat : np.array, lon: np.array, h:
                    np.array, dati:np.array,coord: str="geo"):
        """
        

        Parameters
        ----------
        np.array : lat
            numpy array containing latitudes or magnetic latitudes.
        np.array : lon
            numpy array containing longitudes or magnetic local time.
        np.array : h
            height in km.
        np.array : dati
            np.array containing datetimes.
        coord : TYPE, optional
            Options geo,mlt. The default is "geo".

        Returns
        -------
        np.array: NeMod
            Electron density in 1/cm-3 as array.

        """
        # print("here")
        # print(dati[0])
        if coord=="geo":
            APX=apexpy.Apex(dati[0].year+dati[0].month/12)
            mlat,mlt = APX.convert(lat, lon, "geo","mlt",datetime=np.array(dati))
        if coord=="mlt":
            mlat = lat
            mlt  = lon
        hh = [dati[i].hour for i in range(len(dati))]
        mm = [dati[i].minute for i in range(len(dati))]
        ss = [dati[i].second for i in range(len(dati))]
        hh = np.float64(hh)+np.float64(mm)/60+np.float64(ss)/3600
        NeMod   = np.exp(self.compute_model(mlat,mlt, h, hh))
        return NeMod
        

    