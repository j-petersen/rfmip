import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from read_netcdf import read_netcdf

def main():
    # ty.plots.styles.use(['typhon'])
    ty.plots.styles.use(['typhon', 'typhon-dark'])
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

    data = np.squeeze(pyarts.xml.load('/Users/jpetersen/rare/rfmip/output/rfmip_no_emission/irradiance.xml'))

    toa = data[:, -1, 0]*-1
    
    # data = pyarts.xml.load('/Users/jpetersen/rare/rfmip/output/rfmip/fluxes.xml')
    # toa = data[:, -1, 0]*-1/np.pi
    
    toa = reduce_by_solarangle(toa)
    solar_const.append(toa)

    return solar_const



def plot_solar_const(solar_const):

    tsi = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/total_solar_irradiance.xml')
    sza = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
    
    mask = tuple([sza<87])
    sites = np.arange(100)[mask]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(np.arange(len(solar_const[0]))[mask], (tsi-solar_const[1])[mask], marker='x', label="RRTMG")
    ax.scatter(np.arange(len(solar_const[0]))[mask], (tsi-solar_const[2])[mask], marker='x', label="ARTS")
    ax.scatter(np.arange(len(solar_const[0]))[mask], (tsi-solar_const[0])[mask], marker='x', label="LBLRTM")

    # ax.scatter(np.arange(len(solar_const[0])), solar_const[1]-solar_const[0], marker='x', label="RRTMG-LBLRTM")
    ax.axhline(0, linewidth=0.5)
    ax.set_ylim(-0.009, 0.004)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel('site')
    ax.set_ylabel(r'difference to target tsi / W$\,$m$^{-2}$')

    # ax2 = ax.twinx()
    # add_solar_zenith_angle(ax2)
        
    ax.legend(frameon=False)
    # fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/difference_tsi.png', dpi=200)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/difference_tsi_dark.png', dpi=200)


def reduce_by_solarangle(toa):
    sza = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
    return toa/np.cos(np.deg2rad(sza))


def add_solar_zenith_angle(ax):
    sza = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
    tsi = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/total_solar_irradiance.xml')
    star_distance=1.495978707e11 # 1au
    earth_radius=6.378e6+1e5
    star_distance-=earth_radius #definition in arts distance from TOA to star

    # error = np.array([np.sin(np.deg2rad(a)) if a <=90 else np.NaN for a in sza])


    angles = np.array([a if a <=90 else np.NAN for a in sza])
    dist_error = earth_radius * np.sin(np.deg2rad(angles))

    error = 1-(abs(star_distance**2/(star_distance+dist_error)**2)) * reduce_by_solarangle(tsi)

    ax.scatter(np.arange(len(sza)), error, marker='_', color='w')
    
    ax.set_ylabel(r'distance scaled error /  W$\,$m$^{-2}$')

    print(reduce_by_solarangle(tsi))
    ax.set_ylim(4/3*np.nanmin(error), -1/3*np.nanmin(error))
    # ax.set_ylim(-22.5, 90)
    # ax.invert_yaxis()
    

def readin_nc(filename, fields=None):
    fh = ty.files.NetCDF4()
    if fields is None:
        return fh.read(filename)
    return fh.read(filename, fields)

if __name__ == '__main__':
    main()