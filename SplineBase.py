"""
Created on Mon Nov 14 11:30:27 2022
SplineBase
Get Spline Basis as object

#Changelog:
    ??/12/2022 LS: First running version
    24/05/2023 LS: docstrings added
@author: schreit
"""
import numpy as np
from scipy.interpolate import BSpline
import functools
import scipy.sparse as sparse
import itertools
# from functools import lru_cache,partial

class SplineBase:
    
    def __init__(self, knotVec_lat:np.array,knotVec_lt:np.array,knotVec_h:np.array,degree=3):
        """
        Initialization of spline base in three dimensions

        Parameters
        ----------
        knotVec_lat : np.array
            knotvector in magnetic latitude (-90 to 90).
        knotVec_lt : np.array
            knotvector in magnetic local time (0 to 24).
        knotVec_h : np.array
            knotvector in altitude.

        Returns
        -------
        None.

        """
        self.KV_lat = knotVec_lat
        self.KV_lt  = knotVec_lt
        self.KV_h   = knotVec_h
        self.ni     = len(knotVec_lat)
        self.nj     = len(knotVec_lt)
        self.nk     = len(knotVec_h)
        
        
        self.degree = degree
        self.rng    = rng=int((self.degree-1)/2)
        self.npar   = (self.ni+2*self.rng)*self.nj*(self.nk+2*self.rng) 

    @functools.lru_cache(maxsize=128)
    def get_spline_kn_la(self,i:int):
        """
        get single dimension spline base function number i for magnetic latitude

        Parameters
        ----------
        i : int
            number of the requested spline base function.

        Returns
        -------
        b1 : b sline
            b spline function.

        """
        # degree=3
        t=self.KV_lat
        knot_vector=np.hstack([(np.zeros(self.degree+1)+t[0]),t,(np.zeros(self.degree+1)+t[-1])])
        k=i+self.degree+1
        if self.degree==3:
            sel=knot_vector[[k-2,k-1,k,k+1,k+2]]
        if self.degree==1:
            sel=knot_vector[[k-1,k,k+1]]
        # print(knot_vector)
        # print(sel)
        b1=BSpline.basis_element(sel)
        return b1
    
    @functools.lru_cache(maxsize=128)
    def get_spline_kn_h(self,i:int):
        """
        get single dimension spline base function number i for altitude

        Parameters
        ----------
        i : int
            number of the requested spline base function.

        Returns
        -------
        b1 : b sline
            b spline function.

        """
        # degree=3
        t=self.KV_h
        knot_vector=np.hstack([(np.zeros(self.degree+1)+t[0]),t,(np.zeros(self.degree+1)+t[-1])])
        k=i+self.degree+1
        if self.degree==3:
            sel=knot_vector[[k-2,k-1,k,k+1,k+2]]
        if self.degree==1:
            sel=knot_vector[[k-1,k,k+1]]
        # print(knot_vector)
        # print(sel)
        b1=BSpline.basis_element(sel)
        return b1

    @functools.lru_cache(maxsize=128)
    def get_spline_peri(self,j:int):
        """
        get single dimension spline base function number j for magnetic local time
        (periodic)

        Parameters
        ----------
        j : int
            number of the requested spline base function.

        Returns
        -------
        b2 : b sline
            b spline function.

        """
        KV=self.KV_lt
        if self.degree ==3 :
            ind=np.array([j-2,j-1,j,j+1,j+2])
        if self.degree ==1 :
            ind=np.array([j-1,j,j+1])
            
        sel_lt=ind%self.nj
        # print(KV)
        # print(sel_lt)
        lT=KV[sel_lt]
        lT[ind<0]-=24
        if (ind>=self.nj).sum()>0:
            lT[ind>=self.nj]+=24
            lT-=24
        
        
        b2=BSpline.basis_element(lT)
        return b2

    def get_spline_base(self,lat:np.float64,lon:np.float64,h:np.float64, i:int , j:int, k:int):
        # print(i,j,k)
        """
        

        Parameters
        ----------
        lat : np.float64
            1D vector containing the magnetic latitude of the locations where 
            the base is to be evaluated.
        lon : np.float64
            1D vector containing the magnetic local time of the locations where 
            the base is to be evaluated.
        h : np.float64
            1D vector containing the altitude of the locations where 
            the base is to be evaluated.
        i : int
            number of basis function for latitude.
        j : int
            number of basis function for local time.
        k : int
            number of basis function for altitude.

        Returns
        -------
        out : TYPE
            Values of the three-dimensional tensor product of the basis functions.

        """
    
        b1=self.get_spline_kn_la(i)
        b3=self.get_spline_kn_h(k)
        b2=self.get_spline_peri(j)
        
        out=b1(lat,extrapolate=False)*(b2(lon,extrapolate=False)+b2(lon-24,extrapolate=False))*b3(h,extrapolate=False)
        out=np.nan_to_num(out)
        # out[out<0.0001]=0
        return out


    def get_spline_base_full(self,lat:np.float64,lt:np.float64,h:np.float64):
        """
        

        Parameters
        ----------
        lat : np.float64
            vector containing latitudes where basis elements are to be evaluated.
        lt : np.float64
            vector containing local times where basis elements are to be evaluated.
        h : np.float64
            vector containing altitudes where basis elements are to be evaluated.

        Returns
        -------
        sparse matrix
            Matrix of dimension len(pos)x i*j*k. containing the basis functions value
            at the specific positions (Designmatrix)

        """
        # rng=int((self.degree-1)/2)
        
        B1 = [np.nan_to_num(self.get_spline_kn_la(i)(lat,extrapolate=False)) for i in range(-self.rng,self.ni+self.rng)]
        B2 = [(np.nan_to_num(self.get_spline_peri(j)(lt,extrapolate=False))
                +np.nan_to_num(self.get_spline_peri(j)(lt-24,extrapolate=False))) for j in range(self.nj)]
        B3 = [np.nan_to_num(self.get_spline_kn_h(k)(h,extrapolate=False)) for k in range(-self.rng,self.nk+self.rng)]
        input = ((i,j,k) for i,j,k in itertools.product(range(self.ni+2*self.rng),range(self.nj),range(self.nk+2*self.rng)))
        def splbase(inp:list):
            """
            

            Parameters
            ----------
            inp : list
                list of tupel dimension 3.

            Returns
            -------
            sparse matrix
                values of the basis function's tensor product evaluated at the locations.

            """
            i,j,k=inp
            # print(inp)
            tmp=B1[i]*B2[j]*B3[k]
            return sparse.lil_matrix(tmp)
        # print(input)
        res   = list(map(splbase,input))
        # def:
        # print(list)
        # print(res[0])
        # print("abnt")
        return sparse.csr_matrix(sparse.vstack(res).transpose())
        
    # def get_spline_base_full(self, lat: np.float64, lt: np.float64, h: np.float64):
    #     B1 = np.array([self.get_spline_kn_la(i)(lat, extrapolate=False) for i in range(-1, self.ni + 1)])
    #     B2 = np.array([(self.get_spline_peri(j)(lt % 24, extrapolate=False)) for j in range(self.nj)])
    #     B3 = np.array([self.get_spline_kn_h(k)(h, extrapolate=False) for k in range(-1, self.nk + 1)])
    
    #     # Kronecker product to obtain tensor product
    #     res = sparse.kron(sparse.kron(B1, B2), B3)
        
    #     return sparse.csr_matrix(res)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    SB=SplineBase(np.array([-60,-30,0,30,60]), np.array([0,6,12,18]), np.array([100,350,22000.001]),degree=3)
    x=np.linspace(-90,90,10)
    y=np.linspace(0,24,900)
    z=np.linspace(0,20000,900)
    bla=SB.get_spline_base_full(np.array([0]), np.array([0]), z)
    plt.plot(z,bla.todense())
