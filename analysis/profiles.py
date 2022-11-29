import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from read_data import get_data

def main():
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    models = ['ARTS', 'LBLRTM', 'RRTMG']
    site = 0
    data = get_data(models=models, direction='downward', site=site, lvl=slice(None, None))
    plot_profile(data, site)
    plt.show()


def plot_profile(data, site):
    heights = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/height_levels.xml')
    fig, ax = plt.subplots(figsize=(12, 9))

    for key, value in data.items():
        ax.plot(value, heights[site]/1000, label=key)

    # ax.set_ylim(-0.01, 0.005)
    ax.set_title(f'site {site}')
    ax.set_xlabel(r'irradiance / W$\,$m$^{-2}$')
    ax.set_ylabel('altitude / km')

    # ax2 = ax.twinx()
    # add_solar_zenith_angle(ax2)
        
    ax.legend(frameon=False)
    fig.savefig(f'/Users/jpetersen/rare/rfmip/plots/analysis/profile_site{site}.png', dpi=200)


if __name__ == '__main__':
    main()