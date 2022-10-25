import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from read_netcdf import read_netcdf

def main():
    solar_const = get_data()
    plot_solar_const(solar_const)
    plt.show()


def get_data():
    filelist = [
        '/Users/jpetersen/rare/rfmip/analysis/data/rsd_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc',
        '/Users/jpetersen/rare/rfmip/analysis/data/rsd_Efx_RRTMG-SW-4-02_rad-irf_r1i1p1f1_gn.nc'
    ]
    solar_const = []
    for i, filename in enumerate(filelist):
        data = readin_nc(filename)
        data = data.isel(expt=0)
        if i == 0:
            toa = data.rsd.values[:, 0]
        else:
            toa = data.rsd.values[:, -1]
        solar_const.append(reduce_by_solarangle(toa))

    return solar_const



def plot_solar_const(solar_const):

    tsi = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/total_solar_irradiance.xml')
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    fig, ax = plt.subplots(figsize=(12, 9))
    
    ax.scatter(np.arange(len(solar_const[0])), solar_const[0]-tsi, marker='x', label="LBLRTM")
    ax.scatter(np.arange(len(solar_const[0])), solar_const[1]-tsi, marker='x', label="RRTMG")
    # ax.scatter(np.arange(len(solar_const[0])), solar_const[1]-solar_const[0], marker='x', label="RRTMG-LBLRTM")

    
    ax.set_ylim(-0.01, 0.01)
    ax.set_xlabel('site')
    ax.set_ylabel(r'Solar constant / W$\,$m$^{-2}$')
    
    ax.legend(frameon=False)
    # fig.savefig()


def reduce_by_solarangle(toa):
    sza = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
    return toa/np.cos(np.deg2rad(sza))


def readin_nc(filename, fields=None):
    fh = ty.files.NetCDF4()
    if fields is None:
        return fh.read(filename)
    return fh.read(filename, fields)

if __name__ == '__main__':
    main()