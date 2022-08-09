import os
import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from experiment_setup import read_exp_setup
import helping_functions as hf


def plot_irradiance(heights, irrad, lam_grid, exp_setup, index) -> None:
    ty.plots.styles.use(["typhon", "typhon-dark"])
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    for i, wavelength in enumerate(lam_grid):
        color = hf.rgb4uvvis(wavelength)
        ax.plot(
            irrad[i, :, 0],
            heights / 1e3,
            color=color,
            marker="v",
            label=str(int(np.round(wavelength))) + " nm",
        )
        ax.plot(irrad[i, :, 1], heights / 1e3, color=color, marker="^")

    ax.set_xlabel(r"spectral irradiance / W$\,$m$^{-2}\,$nm$^{-1}$")
    ax.set_ylabel("height / km")
    ax.legend()
    if not os.path.exists(f"{exp_setup.rfmip_path}plots/{exp_setup.name}/"):
        os.mkdir(f"{exp_setup.rfmip_path}plots/{exp_setup.name}/")
    fig.savefig(
        f"{exp_setup.rfmip_path}plots/{exp_setup.name}/diffuse_flux{index}.png", dpi=200
    )
    
    
def convert_units(exp_setup, spectral_grid, irradiance):
    if exp_setup.which_spectral_grid == 'frequency':
        # no conversion necessary
        return spectral_grid, irradiance
    
    elif exp_setup.which_spectral_grid == "wavelength":
        # input is in nm
        f_grid = ty.physics.wavelength2frequency(spectral_grid * 1e-9)[::-1]
        irradiance, spectral_grid = ty.physics.perfrequency2perwavelength(
            irradiance, f_grid
        )
        irradiance *= 1e-9 # convert from per m to per nm
        spectral_grid *= 1e9
        
        return spectral_grid, irradiance


    elif exp_setup.which_spectral_grid == "kayser":
        # input is in 1/cm
        f_grid = ty.physics.wavenumber2frequency(spectral_grid * 1e2)
        irradiance, spectral_grid = ty.physics.perfrequency2perwavenumber(
            irradiance, f_grid
        )
        irradiance *= 1e2 # convert from per 1/m to per 1/cm
        spectral_grid *= 1e-2

        return spectral_grid, irradiance

        
def plot_flux_profiles(exp_setup) -> None:
    selected_data = pyarts.xml.load(
        f"{exp_setup.rfmip_path}output/{exp_setup.name}/selected_spectral_irradiance.xml"
    )  # wavelength, pressure, down-/upward
    selected_heights = pyarts.xml.load(
        f"{exp_setup.rfmip_path}output/{exp_setup.name}/selected_heights.xml"
    )
    spectral_grid = np.linspace(
        exp_setup.spectral_grid["min"],
        exp_setup.spectral_grid["max"],
        exp_setup.spectral_grid["n"],
        endpoint=True,
    )
    
    for i, profile in enumerate(selected_data):
        spectral_grid_converted, irradiance_converted = convert_units(
            exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=profile)

        plot_irradiance(
            selected_heights[i],
            irradiance_converted,
            spectral_grid_converted,
            exp_setup=exp_setup,
            index=i,
        )

    
def plot_olr(exp_setup) -> None:
    data = np.array(pyarts.xml.load(
        f"{exp_setup.rfmip_path}output/{exp_setup.name}/spectral_irradiance.xml"
    ))  # site, wavelength, pressure, 1, 1, down-/upward
    
    spectral_grid = np.linspace(
        exp_setup.spectral_grid["min"],
        exp_setup.spectral_grid["max"],
        exp_setup.spectral_grid["n"],
        endpoint=True,
    )    
    
    spectral_grid_converted, irradiance_converted = convert_units(
        exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data)

    ty.plots.styles.use(["typhon", "typhon-dark"])
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    dat = data[0,:, -1, 0, 0, 1]
    spectral_grid_converted, irradiance_converted = convert_units(
        exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=dat)

    ax.plot(
        spectral_grid_converted[:-1],
        irradiance_converted[:-1],
    )

    ax.set_ylabel(r"spectral irradiance / W$\,$m$^{-2}\,$cm")
    ax.set_xlabel("wavenumber / cm-1")
    if not os.path.exists(f"{exp_setup.rfmip_path}plots/{exp_setup.name}/"):
        os.mkdir(f"{exp_setup.rfmip_path}plots/{exp_setup.name}/")
    fig.savefig(
        f"{exp_setup.rfmip_path}plots/{exp_setup.name}/olr_spectrum.png", dpi=200
    )


def plot_level_spectra(exp_setup):
    data = np.array(pyarts.xml.load(
        f"{exp_setup.rfmip_path}output/{exp_setup.name}/spectral_irradiance.xml"
    ))  # site, wavelength, pressure, 1, 1, down-/upward
    
    spectral_grid = np.linspace(
        exp_setup.spectral_grid["min"],
        exp_setup.spectral_grid["max"],
        exp_setup.spectral_grid["n"],
        endpoint=True,
    )    
    
    spectral_grid_converted, irradiance_converted = convert_units(
        exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=data)

    ty.plots.styles.use(["typhon", "typhon-dark"])
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for lev in range(data[0].shape[1]):
        dat = data[0, :, lev, 0, 0, 1]
        spectral_grid_converted, irradiance_converted = convert_units(
            exp_setup=exp_setup, spectral_grid=spectral_grid, irradiance=dat)

        ax.plot(
            spectral_grid_converted[:-1],
            irradiance_converted[:-1],
        )

    ax.set_ylabel(r"spectral irradiance / W$\,$m$^{-2}\,$cm")
    ax.set_xlabel("wavenumber / cm-1")
    if not os.path.exists(f"{exp_setup.rfmip_path}plots/{exp_setup.name}/"):
        os.mkdir(f"{exp_setup.rfmip_path}plots/{exp_setup.name}/")
    fig.savefig(
        f"{exp_setup.rfmip_path}plots/{exp_setup.name}/spectrum.png", dpi=200
    )

def main():
    exp_setup = read_exp_setup(exp_name='solar_angle', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    plot_flux_profiles(exp_setup=exp_setup)
    # plot_olr(exp_setup=exp_setup)
    # plot_level_spectra(exp_setup=exp_setup)


if __name__ == "__main__":
    main()
