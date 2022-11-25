import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from read_data import get_data

def main():
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    models = ['LBLRTM', 'RRTMG', 'ARTS']
    data = get_data(models=models, direction='downward', lvl=0)
    plot_surface_fluxes(data)
    plt.show()


def plot_surface_fluxes(data):

    fig, ax = plt.subplots(figsize=(12, 9))

    for key, value in data.items():
        ax.scatter(np.arange(len(value)), value, marker='x', label=key)

    # ax.set_ylim(-0.01, 0.005)
    ax.set_xlabel('site')
    ax.set_ylabel(r'irradiance / W$\,$m$^{-2}$')

    # ax2 = ax.twinx()
    # add_solar_zenith_angle(ax2)
        
    ax.legend(frameon=False)
    fig.savefig('/Users/jpetersen/rare/rfmip/plots/analysis/surface_fluxes.png', dpi=200)


if __name__ == '__main__':
    main()