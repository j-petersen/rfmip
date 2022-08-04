""" This file includes the class and methods to read and write experiment setups. """
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
    solar_type: str
    angular_grid: dict
    savename: str = '' #dataclasses.field(init=False)

    def __post_init__(self):
        self.savename = f'{self.rfmip_path}experiment_setups/{self.name}.json'

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
        solar_type='Chose the type of the star. Options are: None, BlackBody, Spectrum, White',
        which_spectral_grid='give the unit for the f_grid. Options are frequency, wavelength or kayser',
        spectral_grid={'min': 'minimum of spectral grid', 'max': 'minimum of spectral grid', 'n': 'number of spectral grid points'},
        species=['select species used for calculation. Select ["all"] to use all species defined in rfmip.'],
        angular_grid={'N_za_grid': 'Number of zenith angles: recommended 20', 'N_aa_grid': 'Number of azimuth angles: recommended 41', 'za_grid_type': 'Zenith angle grid type: linear, linear_mu or double_gauss'}
    )
    dis.save()
    

def new_test_setup():
    exp = ExperimentSetup(
        name='test',
        description='this is a test',
        rfmip_path='/Users/jpetersen/rare/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/Users/jpetersen/rare/',
        solar_type='BlackBody',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 380, 'max': 780, 'n': 12},
        species=['water_vapor', 'ozone', 'carbon_dioxide_GM', 'methane_GM', 'nitrous_oxide_GM'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()


def olr_setup():
    exp = ExperimentSetup(
        name='olr',
        description='goal is to reproduce a olr plot',
        rfmip_path='/Users/froemer/Documents/wv_continuum/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/Users/froemer/Documents/',
        solar_type='None',
        which_spectral_grid='kayser',
        spectral_grid={'min': 1, 'max': 2500, 'n': 100},
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
        solar_type='Spectrum',
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
        rfmip_path='/Users/jpetersen/rare/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/Users/jpetersen/rare/',
        solar_type='Spectrum',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 200, 'max': 2_500, 'n': 800},
        species=['all'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    exp.save()

def main():
    new_test_setup()
    # exp_setup_description()
    olr_setup()
    solar_angle_dependency_setup()
    rfmip_setup()


if __name__ == '__main__':
    main()
