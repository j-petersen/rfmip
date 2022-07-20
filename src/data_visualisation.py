import os
import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from experiment_setup import read_exp_setup
from helping_funcs import conversions as con

def plot_irradiance(irrad, lam_grid, exp_setup) -> None:
    ty.plots.styles.use(["typhon", "typhon-dark"])
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    for i, wavelength in enumerate(lam_grid):
        # color = con.wavelength2rgb(wavelength)
        color = con.rgb4uvvis(wavelength)
        ax.plot(irrad[i, :, 0], np.arange(np.shape(irrad)[1]), color=color, marker='v', label=str(int(np.round(wavelength)))+' nm')
        ax.plot(irrad[i, :, 1], np.arange(np.shape(irrad)[1]), color=color, marker='^')

    ax.set_xlabel(r'spectral irradiance / W$\,$m$^{-2}\,$nm$^{-1}$')
    ax.set_ylabel('layer / 1')
    ax.legend()
    if not os.path.exists(f'{exp_setup.rfmip_path}plots/{exp_setup.name}/'):
            os.mkdir(f'{exp_setup.rfmip_path}plots/{exp_setup.name}/')
    fig.savefig(f'{exp_setup.rfmip_path}plots/{exp_setup.name}/diffuse_flux.png', dpi=200)
    plt.show()

def main() -> None:
    exp_setup = read_exp_setup(exp_name='test')
    data = pyarts.xml.load(f'{exp_setup.rfmip_path}output/{exp_setup.name}/combined_spectral_irradiance.xml') # wavelength, pressure, down-/upward
    lam_grid = np.linspace(exp_setup.lam_grid['min_lam'], exp_setup.lam_grid['max_lam'], exp_setup.lam_grid['nlam'], endpoint=True)*1e-9
    f_grid = ty.physics.wavelength2frequency(lam_grid)
    irrad_diffuse_lam, lam_grid = ty.physics.perfrequency2perwavelength(data, f_grid)
    irrad_diffuse_nm = irrad_diffuse_lam*1e-9
    lam_grid_nm = lam_grid*1e9
    plot_irradiance(irrad_diffuse_nm, lam_grid_nm, exp_setup=exp_setup)

if __name__ == '__main__':
    main()