import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from read_data import get_data

def main():
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    models = ['ARTS', 'LBLRTM']
    sza = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')
    mask = tuple([(84 < sza) & (sza < 90)])
    sites = np.arange(100)[mask]
    # sites = np.append(sites, 75)
    exp='rfmip_lvl'
    data_down = get_data(models=models, direction='downward', experiment=exp, site=sites, lvl=slice(None, None))
    data_up = get_data(models=models, direction='up', experiment=exp, site=sites, lvl=slice(None, None))
    plot_profiles_sza(data_down, data_up, sites)
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


def plot_profiles_sza(data_down, data_up, sites):
    heights = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/height_levels.xml')
    sza = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/solar_zenith_angle.xml')

    fig, ax = plt.subplots(figsize=(12, 9))
    for key, value in data_down.items():
        if key != 'LBLRTM':
            if len(value[0]) == 121:
                value = value[:, ::2]
                data_up[key] = data_up[key][:, ::2]
            rel_diff_down = rel_error(data_down['LBLRTM'], value)
            rel_diff_up = rel_error(data_up['LBLRTM'], data_up[key])
            for i, site in enumerate(sites):
                if not np.isinf(rel_diff_down[i]).any():
                    line = ax.plot(rel_diff_down[i,:-3], heights[site, :-3]/1000, label=f'site: {site}')#, label=f'{np.round(sza[site], 1)} (site: {site})')
                    ax.plot(rel_diff_up[i, :-3], heights[site, :-3]/1000, color=line[0].get_color(), linestyle='--')
                    ax.plot(rel_diff_up[i, -4], heights[site, -4]/1000, marker='^', color=line[0].get_color())
        
    ax.plot(rel_diff_down[i, -4], heights[site, -4]/1000, marker='v', color=line[0].get_color())
    
    ax.axvline(0, linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlabel(r'relative difference / %', fontsize=24)
    ax.set_ylabel('altitude / km', fontsize=24)
        
    ax.legend(frameon=False)
    fig.savefig(f'/Users/jpetersen/rare/rfmip/plots/analysis/profiles_up_down_high_sza_dark.png', dpi=200)


def rel_error(reference, value):
    return (value - reference) / reference * 100


if __name__ == '__main__':
    main()