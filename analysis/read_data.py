import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt


def main():
    pass

def filename_of_model(model, direction='downward', experiment='rfmip_no_emission'):
    if model == 'LBLRTM':
        if direction == 'downward':
            f = '/Users/jpetersen/rare/rfmip/analysis/data/rsd_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc'
        else:
            f = '/Users/jpetersen/rare/rfmip/analysis/data/rsu_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc'
    elif model == 'RRTMG':
        if direction == 'downward':
            f = '/Users/jpetersen/rare/rfmip/analysis/data/rsd_Efx_RRTMG-SW-4-02_rad-irf_r1i1p1f1_gn.nc'
        else:
            f = '/Users/jpetersen/rare/rfmip/analysis/data/rsu_Efx_RRTMG-SW-4-02_rad-irf_r1i1p1f1_gn.nc'
    elif model == 'ARTS':
        f = f'/Users/jpetersen/rare/rfmip/output/{experiment}/irradiance.xml'
    else: 
        print("There is not such a model!")
    return f

def get_data(models, direction='downward', experiment='rfmip_no_emission', lvl=0, site=slice(0, 100, None)):
    if type(lvl) == int:
        lvl = slice(lvl, lvl+1, None) 
    data_dict = {}
    for model in models:
        filename = filename_of_model(model, direction, experiment=experiment)
        if model != 'ARTS':
            data = readin_nc(filename)
            data = data.isel(expt=0)
            if direction == 'downward':
                irrad = data.rsd.values
            else:
                irrad = data.rsu.values

            if model == 'LBLRTM':
                irrad = irrad[:, ::-1]
        else: 
            data = np.squeeze(pyarts.xml.load(filename))
            irrad = data[:, :, 0]*-1 if direction == 'downward' else data[:, :, 1]
            irrad = np.array(irrad)

        data_dict[model] = np.squeeze(irrad[site, lvl])

    return data_dict


def readin_nc(filename, fields=None):
    fh = ty.files.NetCDF4()
    if fields is None:
        return fh.read(filename)
    return fh.read(filename, fields)


if __name__ == '__main__':
    main()