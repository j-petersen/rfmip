""" This file includes the class and methods to read and write experiment setups. """
import os
import numpy as np
import dataclasses, json

@dataclasses.dataclass()
class ExperimentSetup:
    """Class for keeping track of the experiment setup."""
    name: str
    # _: dataclasses.KW_ONLY # enter data as kw atfer this
    description: str
    which_spectral_grid: str
    spectral_grid: dict
    species: list
    rfmip_path: str
    input_folder: str
    arts_data_path: str
    lookuptable: str
    solar_type: str
    planck_emission: str
    angular_grid: dict
    savename: str = '' #dataclasses.field(init=False)

    def __post_init__(self):
        path = f'{os.getcwd()}/experiment_setups/'
        if not os.path.exists(path):
            raise FileNotFoundError
        self.savename = f'{path}{self.name}.json'

    def __repr__(self):
        out_str = 'ExperimentSetup:\n'
        for key, value in self.__dict__.items():
            out_str += f'   {key}: {value}\n'
        return out_str

    def save(self):
        with open(self.savename, 'w') as fp:
            json.dump(self, fp, cls=EnhancedJSONEncoder, indent=True)


class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)


def read_exp_setup(exp_name, path) -> dataclasses.dataclass:
    with open(f'{path}{exp_name}.json') as fp:
        d = json.load(fp)
        exp = ExperimentSetup(**d)
    return exp


def exp_setup_description():
    dis = ExperimentSetup('description',
        description='This is the discription of the Experiment Setup class variables.',
        rfmip_path='path to the rfmip directory.',
        input_folder='relative path from the rfmip dir to the input data for the run.',
        arts_data_path='Path to the directory where arts cat (arts-cat-data) and xml (arts-xml-data) data are.',
        lookuptable='Name of the lookuptable for the calculation',
        solar_type='Chose the type of the star. Options are: None, BlackBody, Spectrum, White',
        planck_emission='Toggle planck emission on (set 1) or off (set 0)',
        which_spectral_grid='give the unit for the f_grid. Options are frequency, wavelength or kayser',
        spectral_grid={'min': 'minimum of spectral grid', 'max': 'minimum of spectral grid', 'n': 'number of spectral grid points'},
        species=['select species used for calculation. Select ["all"] to use all species defined in rfmip.'],
        angular_grid={'N_za_grid': 'Number of zenith angles: recommended 20', 'N_aa_grid': 'Number of azimuth angles: recommended 41', 'za_grid_type': 'Zenith angle grid type: linear, linear_mu or double_gauss'}
    )
    dis.save()
    
def test_setup():
    exp = ExperimentSetup(
        name='test',
        description='simple test case',
        rfmip_path='/Users/jpetersen/rare/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/Users/jpetersen/rare/',
        lookuptable='test.xml',
        solar_type='BlackBody',
        planck_emission='1',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 380, 'max': 780, 'n': 21},
        species=['water_vapor', 'ozone', 'carbon_dioxide_GM', 'nitrous_oxide_GM'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()

def testing_rfmip_setup():
    exp = ExperimentSetup(
        name='testing_rfmip',
        description='local testing case for rfmip',
        rfmip_path='/Users/jpetersen/rare/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/Users/jpetersen/rare/',
        lookuptable='test_rfmip.xml',
        solar_type='Spectrum',
        planck_emission='1',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 115.5, 'max': 9_999.5, 'n': 2**10},
        species=['all'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()


def olr_setup():
    exp = ExperimentSetup(
        name='olr',
        description='goal is to reproduce a olr plot',
        rfmip_path='/Users/jpetersen/rare/rfmip/',
        input_folder='input/olr/',
        arts_data_path='/Users/jpetersen/rare/',
        lookuptable='olr.xml',
        solar_type='None',
        planck_emission='1',
        which_spectral_grid='kayser',
        spectral_grid={'min': 1, 'max': 2500, 'n': 1000},
        species=['water_vapor', 'ozone', 'carbon_dioxide_GM', 'nitrous_oxide_GM'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()

def solar_angle_dependency_setup():
    exp = ExperimentSetup(
        name='solar_angle',
        description='Test to investigate the dependency of the solar angle',
        rfmip_path='/Users/jpetersen/rare/rfmip/',
        input_folder='input/solar_angle/',
        arts_data_path='/Users/jpetersen/rare/',
        lookuptable='solar_angle.xml',
        solar_type='Spectrum',
        planck_emission='1',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 380, 'max': 780, 'n': 12},
        species=['water_vapor', 'ozone', 'carbon_dioxide_GM', 'methane_GM', 'nitrous_oxide_GM'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()

def rfmip_setup():
    exp = ExperimentSetup(
        name='rfmip',
        description='rfmip',
        rfmip_path='/work/um0878/users/jpetersen/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/work/um0878/users/jpetersen/',
        lookuptable='rfmip.xml',
        solar_type='Spectrum',
        planck_emission='1',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 115.5, 'max': 9_999.5, 'n': 2**15},
        species=['all'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()

def rfmip_no_star_setup():
    exp = ExperimentSetup(
        name='rfmip_no_star',
        description='rfmip but without a star. (check the effect of planck emission on the flux)',
        rfmip_path='/work/um0878/users/jpetersen/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/work/um0878/users/jpetersen/',
        lookuptable='rfmip.xml',
        solar_type='None',
        planck_emission='1',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 115.5, 'max': 9_999.5, 'n': 2**15},
        species=['all'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()

def rfmip_no_emission_setup():
    exp = ExperimentSetup(
        name='rfmip_no_emission',
        description='rfmip but without thermal emission in the atmoshere.',
        rfmip_path='/work/um0878/users/jpetersen/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/work/um0878/users/jpetersen/',
        lookuptable='rfmip.xml',
        solar_type='Spectrum',
        planck_emission='0',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 115.5, 'max': 9_999.5, 'n': 2**15},
        species=['all'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()

def main():
    exp_setup_description()
    test_setup()
    testing_rfmip_setup()
    olr_setup()
    solar_angle_dependency_setup()
    rfmip_setup()
    rfmip_no_star_setup()
    rfmip_no_emission_setup()


if __name__ == '__main__':
    main()
