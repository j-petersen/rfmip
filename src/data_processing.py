import pyarts
import numpy as np
import matplotlib.pyplot as plt

from experiment_setup import read_exp_setup

def read_spectral_irradiance(exp_setup) -> np.ndarray:
    data = np.squeeze(pyarts.xml.load(f'{exp_setup.rfmip_path}output/{exp_setup.name}/spectral_irradiance.xml')) # site, wavelength, pressure, down-/upward
    return data

def combine_sites(data, exp_setup) -> np.ndarray:
    weights = pyarts.xml.load(f'{exp_setup.rfmip_path}input/profil_weight.xml')
    data = np.average(data, axis=0, weights=weights)
    return data
    

def save_data(data, exp_setup) -> None:
    pyarts.xml.save(data, f'{exp_setup.rfmip_path}output/{exp_setup.name}/combined_spectral_irradiance.xml')


def main() -> None:
    exp_setup = read_exp_setup(exp_name='test')
    data = read_spectral_irradiance(exp_setup)
    data = combine_sites(data, exp_setup)
    save_data(data, exp_setup)


if __name__ == '__main__':
    main()