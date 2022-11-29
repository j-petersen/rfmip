import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from read_data import get_data

def main():
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    models = ['LBLRTM', 'RRTMG', 'ARTS']
    data = get_data(models=models, direction='upward', lvl=0)
    plot_surface_upward_LBLRTM(data)
    data = get_data(models=models, direction='upward', lvl=60)
    plot_upward_TOA_LBLRTM(data)
    plt.show()


def plot_upward_TOA_LBLRTM(data):
    fig, ax = plt.subplots(figsize=(12, 9))
    print(np.shape(data['LBLRTM']))

    for key, value in data.items():
        if key != 'LBLRTM':
            ax.scatter(np.arange(len(value)), data['LBLRTM']-value, marker='x', label=key)

    ax.set_ylim(-6, 1)
    ax.set_xlabel('site')
    ax.set_ylabel(r'irradiance / W$\,$m$^{-2}$')
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/upward_TOA_fluxes_LBLRTM.png', dpi=200)


def plot_surface_upward_LBLRTM(data):
    fig, ax = plt.subplots(figsize=(12, 9))

    for key, value in data.items():
        if key != 'LBLRTM':
            ax.scatter(np.arange(len(value)), data['LBLRTM']-value, marker='x', label=key)

    ax.set_ylim(-5, 1)
    ax.set_xlabel('site')
    ax.set_ylabel(r'irradiance / W$\,$m$^{-2}$')
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/upward_surface_fluxes_LBLRTM.png', dpi=200)


if __name__ == '__main__':
    main()