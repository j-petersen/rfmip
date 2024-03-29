import pyarts
import numpy as np
import typhon as ty
import matplotlib as mpl
import matplotlib.pyplot as plt

from read_data import get_data

def main():
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    models = ['LBLRTM', 'RRTMG', 'ARTS']
    data = get_data(models=models, direction='downward', experiment='rfmip_no_emission', lvl=0)
    # plot_surface_fluxes(data)
    # plot_surface_flux_LBLRTM(data)
    # quantify_differences(data)
    # correlation_var(data, ['solar_zenith_angle', None], 'deg')
    # correlation_var(data, ['surface_temperature', None], 'K')
    # correlation_var(data, ['relative_path_length', None], '1')
    # correlation_var(data, ['integrated_water_vapor', get_iwv()], r'kg$\,$m$^{-2}$')
    # lat = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/site_pos.xml')[:, 0]
    # correlation_var(data, ['latitude', lat], 'deg')
    # cross_correlation(data, variable1=['solar_zenith_angle', None], variable2=['integrated_water_vapor', get_iwv()], units=['deg', r'kg$\,$m$^{-2}$'])
    # cross_correlation(data, variable1=['relative_path_length', None], variable2=['integrated_water_vapor', get_iwv()], units=['1', r'kg$\,$m$^{-2}$'])
    # cross_correlation(data, variable1=['solar_zenith_angle', None], variable2=['total_irradiace', data['LBLRTM']], units=['deg', r"W$\,$m$^{-2}$"])
    # cross_correlation(data, variable1=['solar_zenith_angle', None], variable2=['relative_path_length', None], units=['deg', "1"])
    # data = get_data(models=models, direction='upward', lvl=0)
    plot_surface_upward_LBLRTM(data)

    plt.show()


def plot_surface_fluxes(data):
    fig, ax = plt.subplots(figsize=(12, 9))

    for key, value in data.items():
        ax.scatter(np.arange(len(value)), value, marker='x', label=key)

    ax.set_xlabel('site')
    ax.set_ylabel(r'irradiance / W$\,$m$^{-2}$')
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/surface_fluxes.png', dpi=200)


def plot_surface_flux_LBLRTM(data):
    fig, ax = plt.subplots(figsize=(12, 9))

    for key, value in data.items():
        if key != 'LBLRTM':
            mask = tuple([data['LBLRTM']!=0])
            ax.scatter(np.arange(len(value))[mask], rel_error(data['LBLRTM'], value)[mask], marker='x', label=key)

    ax.axhline(0, linewidth=0.5)
    ax.set_ylim(-0.5, 7)
    ax.set_xlabel('site')
    ax.set_ylabel(r'relative difference / %')
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/surface_fluxes_LBLRTM_dark.png', dpi=200)

def quantify_differences(data):
    mask = tuple([data['LBLRTM']!=0])
    relerr = rel_error(data['LBLRTM'], data['ARTS'])[mask]
    print(f'ARTS relative mean: {np.mean(relerr)}, std: {np.std(relerr)}')
    abserr = (data['LBLRTM']- data['ARTS'])[mask]
    print(f'ARTS absolute mean: {np.mean(abserr)}, std: {np.std(abserr)}')

    err = rel_error(data['LBLRTM'], data['RRTMG'])[mask]
    print(f'RRTMG mean: {np.mean(err)}, std: {np.std(err)}')
    abserr = (data['LBLRTM']- data['RRTMG'])[mask]
    print(f'RRTMG absolute mean: {np.mean(abserr)}, std: {np.std(abserr)}')

    sza = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
    sza_mask = tuple([sza<=60])
    relerr = rel_error(data['LBLRTM'], data['ARTS'])[sza_mask]
    print('with SZA< 60 deg')
    print(f'ARTS relative mean: {np.mean(relerr)}, std: {np.std(relerr)}')
    abserr = (data['LBLRTM']- data['ARTS'])[sza_mask]
    print(f'ARTS absolute mean: {np.mean(abserr)}, std: {np.std(abserr)}')


def plot_surface_upward_LBLRTM(data):
    fig, ax = plt.subplots(figsize=(12, 9))

    for key, value in data.items():
        if key != 'LBLRTM':
            ax.scatter(np.arange(len(value)), rel_error(data['LBLRTM'], value), marker='x', label=key)

    ax.set_ylim(-5, 1)
    ax.set_xlabel('site')
    ax.set_ylabel(r'irradiance / W$\,$m$^{-2}$')
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/surface_fluxes_LBLRTM.png', dpi=200)


def correlation_var(data, variable=['solar_zenith_angle', None], unit='deg'):
    var = variable[1]
    if var is None:
        if variable[0] == 'relative_path_length':
            sza = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
            var = (1/np.cos(np.deg2rad(sza)))
        else:
            var = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/{variable[0]}.xml')

    fig, ax = plt.subplots(figsize=(12, 9))
    for key, value in data.items():
        if key != 'LBLRTM':
            mask = tuple([data['LBLRTM']!=0])
            scat = ax.scatter(var[mask], rel_error(data['LBLRTM'], value)[mask], marker='x', label=key)
            # linear fit 
            m, b = np.polyfit(var[mask], rel_error(data['LBLRTM'], value)[mask], 1)
            x = np.linspace((var[mask]).min()*0.9, (var[mask]).max()*1.05, 2)
            ax.plot(x, b+m*x, linestyle='--', color=scat.get_facecolor()[0])
            
            mse = np.square(np.subtract(rel_error(data['LBLRTM'], value)[mask], b+m*var[mask])).mean() 
            rmse = np.sqrt(mse)
            print(f"{key} - Root Mean Square Error: {rmse}")
    
    if variable[0] == 'relative_path_length':
        ax.axvline(1/np.cos(np.deg2rad(60)), linewidth=0.5, c='w')
        


    ax.axhline(0, linewidth=0.5, c='w')

    ax.set_xlabel(f'{variable[0].replace("_", " ")} / {unit}')
    ax.set_ylabel(r'relative difference / %')
        
    ax.legend(frameon=False, loc=(0.12, 0.9))
    fig.savefig(f'/Users/jpetersen/rare/rfmip/plots/analysis/corr_surface_fluxes_{variable[0]}_LBLRTM_dark.png', dpi=200)


def cross_correlation(data, variable1=['solar_zenith_angle', None], variable2=['surface_temperature', None], units=['deg', 'K']):
    var1 = variable1[1]
    var2 = variable2[1]

    if var1 is None:
        if variable1[0] == 'relative_path_length':
            sza = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
            var1 = (1/np.cos(np.deg2rad(sza)))
        else:
            var1 = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/{variable1[0]}.xml')
    if var2 is None:
        if variable2[0] == 'relative_path_length':
            sza = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
            var2 = (1/np.cos(np.deg2rad(sza)))
        else:
            var2 = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/{variable2[0]}.xml')

    fig, ax = plt.subplots(figsize=(12, 9))

    mask = tuple([data['LBLRTM']!=0])
    rel_diff = rel_error(data['LBLRTM'], data['ARTS'])
    rel_diff[rel_diff == np.inf] = np.nan

    # cmap = plt.cm.get_cmap("inferno_r").copy()
    cmap = plt.cm.get_cmap("autumn_r").copy()
    norm = mpl.colors.Normalize(vmin=np.nanmin(rel_diff), vmax=np.nanmax(rel_diff))

    for site in np.arange(len(var1))[mask]:
        sca = ax.scatter(var1[site], var2[site], c=rel_diff[site], marker='x', cmap=cmap, norm=norm)
    
    fig.colorbar(
        sca,
        ax=ax,
        extend="neither",
        label=r'relative difference / %',
    )

    ax.set_xlabel(f'{variable1[0].replace("_", " ")} / {units[0]}')
    ax.set_ylabel(f'{variable2[0].replace("_", " ")} / {units[1]}')

    fig.savefig(f'/Users/jpetersen/rare/rfmip/plots/analysis/cross_corr_surface_fluxes_{variable1[0]}_{variable2[0]}_LBLRTM_dark.png', dpi=200)


def get_iwv():
    atm_fields = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/atm_fields.xml')
    sites = 100
    iwv = np.empty((sites))
    for site in range(sites):
        p_field = np.squeeze(atm_fields[site].grids[1])
        t_field = np.squeeze(atm_fields[site][0, :])
        z_field = np.squeeze(atm_fields[site][1, :])
        water_vapor = np.squeeze(atm_fields[site][4, :])
        ozone = np.squeeze(atm_fields[site][5, :])

        iwv[site] = ty.physics.integrate_water_vapor(water_vapor, p_field, T=t_field, z=z_field)
    return iwv


def rel_error(reference, value):
    return (value - reference) / reference * 100


if __name__ == '__main__':
    main()