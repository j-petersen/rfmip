import pyarts
import numpy as np
import typhon as ty
import matplotlib as mpl
import matplotlib.pyplot as plt

from read_data import get_data

def main():
    ty.plots.styles.use(['typhon'])
    models = ['LBLRTM', 'RRTMG', 'ARTS']
    data_down = get_data(models=models, direction='downward', experiment='rfmip_no_emission', lvl=0)
    data_up = get_data(models=models, direction='upward', experiment='rfmip_no_emission', lvl=60)
    correlation_var(data_down, data_up, ['solar_zenith_angle', None], 'deg')

    plt.show()


def plot_surface_fluxes(data):
    fig, ax = plt.subplots(figsize=(12, 9))

    for key, value in data.items():
        ax.scatter(np.arange(len(value)), value, marker='x', label=key)

    ax.set_xlabel('site')
    ax.set_ylabel(r'irradiance / W$\,$m$^{-2}$')
        
    ax.legend(frameon=False)
    fig.savefig(None, dpi=200)


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
    fig.savefig(None, dpi=200)


def correlation_var(data_down, data_up, variable=['solar_zenith_angle', None], unit='deg'):
    var = variable[1]
    if var is None:
        var = pyarts.xml.load(f'/Users/jpetersen/rare/rfmip/input/rfmip/{variable[0]}.xml')

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = tuple([data_down['LBLRTM']!=0])
    ax.scatter(var[mask], rel_error(data_down['LBLRTM'], data_down['ARTS'])[mask], marker='x', label='down')
    ax.scatter(var[mask], rel_error(data_up['LBLRTM'], data_up['ARTS'])[mask], marker='x', label='up')
    ax.axhline(0, linewidth=0.5)

    ax.set_xlabel(f'{variable[0].replace("_", " ")} / {unit}')
    ax.set_ylabel(r'relative difference / %')
        
    ax.legend(frameon=False)
    fig.savefig(f'/Users/jpetersen/rare/rfmip/plots/analysis/diff_down_up{variable[0]}_LBLRTM.png', dpi=200)


def rel_error(reference, value):
    return (value - reference) / reference * 100


if __name__ == '__main__':
    main()