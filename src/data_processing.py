import pyarts
import numpy as np

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
            f"{exp_setup.rfmip_path}input/{exp_setup.name}/additional_data/heights.xml"
        )
    )
    return data


def combine_sites(data, exp_setup) -> np.ndarray:
    weights = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}profil_weight.xml')
    data = np.average(data, axis=0, weights=weights)
    return data


def save_data(data, exp_setup, name) -> None:
    pyarts.xml.save(data, f"{exp_setup.rfmip_path}output/{exp_setup.name}/{name}.xml")


def main() -> None:
    exp_setup = read_exp_setup(exp_name='test', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    data = read_spectral_irradiance(exp_setup)
    heights = read_heights(exp_setup)
    combined_data = combine_sites(data, exp_setup)

    select = [20, 29, 39]  # select profiles for polar, mid-latitudes, and tropics
    selected_data, selected_heigths = data[select], heights[select]

    save_data(combined_data, exp_setup, "combined_spectral_irradiance")
    save_data(selected_data, exp_setup, "selected_spectral_irradiance")
    save_data(selected_heigths, exp_setup, "selected_heights")


if __name__ == "__main__":
    main()
