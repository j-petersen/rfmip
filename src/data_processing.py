import pyarts
import numpy as np
import typhon as ty

from experiment_setup import read_exp_setup


def read_spectral_irradiance(exp_setup) -> np.ndarray:
    data = np.squeeze(
        pyarts.xml.load(
            f"{exp_setup.rfmip_path}output/{exp_setup.name}/spectral_irradiance.xml"
        )
    )  # site, wavelength, pressure, down-/upward
    return data


def read_heights(exp_setup) -> np.ndarray:
    data = np.squeeze(
        pyarts.xml.load(
            f"{exp_setup.rfmip_path}{exp_setup.input_folder}heights.xml"
        )
    )
    return data


def combine_sites(data, exp_setup) -> np.ndarray:
    weights = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}profil_weight.xml')
    data = np.average(data, axis=0, weights=weights)
    return data


def save_data(data, exp_setup, name) -> None:
    pyarts.xml.save(data, f"{exp_setup.rfmip_path}output/{exp_setup.name}/{name}.xml")


def calc_flux(exp_setup) -> None:
    spec_irrad = read_spectral_irradiance(exp_setup)
    f_grid = f_grid_from_spectral_grid(exp_setup)

    irrad = np.trapz(spec_irrad, f_grid, axis=1)
    save_data(irrad, exp_setup, 'irradiance')


def f_grid_from_spectral_grid(exp_setup):
    if exp_setup.which_spectral_grid == 'frequency':
        f_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)
    elif exp_setup.which_spectral_grid == 'wavelength':
        lam_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)*1e-9
        f_grid = ty.physics.wavelength2frequency(lam_grid)[::-1]
    elif exp_setup.which_spectral_grid == 'kayser':
        kayser_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)*1e2
        f_grid = ty.physics.wavenumber2frequency(kayser_grid)

    return f_grid


def main() -> None:
    exp_setup = read_exp_setup(exp_name='rfmip_lvl', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    exp_setup.rfmip_path = '/Users/jpetersen/rare/rfmip/'
    exp_setup.arts_data_path = '/Users/jpetersen/rare/'
    calc_flux(exp_setup)
    # data = read_spectral_irradiance(exp_setup)
    # heights = read_heights(exp_setup)
    # combined_data = combine_sites(data, exp_setup)

    # select = [20, 29, 39]  # select profiles for polar, mid-latitudes, and tropics
    # selected_data, selected_heigths = data[select], heights[select]

    # save_data(combined_data, exp_setup, "combined_spectral_irradiance")
    # save_data(selected_data, exp_setup, "selected_spectral_irradiance")
    # save_data(selected_heigths, exp_setup, "selected_heights")


if __name__ == "__main__":
    main()
