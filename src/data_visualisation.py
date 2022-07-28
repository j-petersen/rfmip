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
    plt.show()


def plot_flux_profiles(exp_setup) -> None:
    combined_data = pyarts.xml.load(
        f"{exp_setup.rfmip_path}output/{exp_setup.name}/combined_spectral_irradiance.xml"
    )  # wavelength, pressure, down-/upward
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

    if exp_setup.which_spectral_grid == "wavelength":
        spectral_grid_nm = spectral_grid  # input is in nm
        f_grid = ty.physics.wavelength2frequency(spectral_grid_nm * 1e-9)[::-1]
        for i, profile in enumerate(selected_data):
            irrad_diffuse, spectral_grid = ty.physics.perfrequency2perwavelength(
                profile, f_grid
            )
            irrad_diffuse_nm = irrad_diffuse * 1e-9

            plot_irradiance(
                selected_heights[i],
                irrad_diffuse_nm,
                spectral_grid_nm,
                exp_setup=exp_setup,
                index=i,
            )

    elif exp_setup.which_spectral_grid == "frequency":
        for i, profile in enumerate(selected_data):
            plot_irradiance(
                selected_heights[i],
                profile,
                spectral_grid,
                exp_setup=exp_setup,
                index=i,
            )

    elif exp_setup.which_spectral_grid == "kayser":
        spectral_grid_cm = spectral_grid  # input is in cm
        f_grid = ty.physics.wavenumber2frequency(spectral_grid_cm * 1e2)
        for i, profile in enumerate(selected_data):
            irrad_diffuse, spectral_grid = ty.physics.perfrequency2perwavenumber(
                profile, f_grid
            )
            irrad_diffuse_cm = irrad_diffuse * 1e2

            plot_irradiance(
                selected_heights[i],
                irrad_diffuse_cm,
                spectral_grid_cm,
                exp_setup=exp_setup,
                index=i,
            )


def main():
    exp_setup = read_exp_setup(exp_name='test', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    plot_flux_profiles(exp_setup=exp_setup)


if __name__ == "__main__":
    main()
