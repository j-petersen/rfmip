import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from read_data import get_data

def main():
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    models = ['LBLRTM', 'RRTMG', 'ARTS']
    # data = get_data(models=models, direction='upward', lvl=0)
    # plot_surface_upward_LBLRTM(data)
    data = get_data(models=models, direction='upward', lvl=60)
    plot_upward_TOA_LBLRTM(data)
    quantify_differences(data)
    plt.show()


def plot_upward_TOA_LBLRTM(data):
    fig, ax = plt.subplots(figsize=(12, 9))
    print(np.shape(data['LBLRTM']))

    for key, value in data.items():
        if key != 'LBLRTM':
            mask = tuple([data['LBLRTM']!=0])
            ax.scatter(np.arange(len(value))[mask], rel_error(data['LBLRTM'], value)[mask], marker='x', label=key)

    ax.axhline(0, linewidth=0.5)
    ax.set_ylim(-0.5, 7)
    ax.set_xlabel('site')
    ax.set_ylabel(r'relative difference / %')
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/upward_TOA_fluxes_LBLRTM_dark.png', dpi=200)

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
            mask = tuple([data['LBLRTM']!=0])
            ax.scatter(np.arange(len(value))[mask], rel_error(data['LBLRTM'], value)[mask], marker='x', label=key)
    
    ax.axhline(0, linewidth=0.5)
    ax.set_xlabel('site')
    ax.set_ylabel(r'relative difference / %')
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/upward_surface_fluxes_LBLRTM.png', dpi=200)


def rel_error(reference, value):
    return (value - reference) / reference * 100


if __name__ == '__main__':
    main()