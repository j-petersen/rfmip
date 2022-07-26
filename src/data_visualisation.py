# %%
import os
import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from experiment_setup import read_exp_setup
import helping_functions as hf

def plot_irradiance(irrad, lam_grid, exp_setup) -> None:
    ty.plots.styles.use(["typhon", "typhon-dark"])
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    for i, wavelength in enumerate(lam_grid):
        # color = hf.wavelength2rgb(wavelength)
        color = hf.rgb4uvvis(wavelength)
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
    data = pyarts.xml.load(
        f'{exp_setup.rfmip_path}output/{exp_setup.name}/combined_spectral_irradiance.xml') # wavelength, pressure, down-/upward
    spectral_grid = np.linspace(
        exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], 
        exp_setup.spectral_grid['n'], endpoint=True)
    
    if exp_setup.which_spectral_grid == 'wavelength':
        f_grid = ty.physics.wavelength2frequency(spectral_grid*1e-9)[::-1]
        irrad_diffuse, spectral_grid = ty.physics.perfrequency2perwavelength(data, f_grid)
        irrad_diffuse_nm = irrad_diffuse*1e-9
        spectral_grid_nm = spectral_grid*1e9
        
        plot_irradiance(irrad_diffuse_nm, spectral_grid_nm, exp_setup=exp_setup)

    elif exp_setup.which_spectral_grid == 'frequency':
        plot_irradiance(data, spectral_grid, exp_setup=exp_setup)

    elif exp_setup.which_spectral_grid == 'kayser':
        f_grid = ty.physics.wavenumber2frequency(spectral_grid*1e2)
        irrad_diffuse, spectral_grid = ty.physics.perfrequency2perwavenumber(data, f_grid)
        irrad_diffuse_cm = irrad_diffuse*1e2
        spectral_grid_cm = spectral_grid*1e-2
        
        plot_irradiance(irrad_diffuse_cm, spectral_grid_cm, exp_setup=exp_setup)


if __name__ == '__main__':
    main()
# %%
