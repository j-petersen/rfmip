import os
import pyarts
import numpy as np
import typhon as ty
from itertools import repeat
from multiprocessing import Pool

import write_xml_input_data as input_data
from experiment_setup import ExperimentSetup


def calc_lookup(exp_setup, recalculate=False):
    lut = BatchLookUpTable(exp_setup)
    lut.calculate(recalculate=recalculate)

class BatchLookUpTable():
    def __init__(self, exp_setup, ws=None):
        self.exp_setup = exp_setup
        self.new_ws = False
        if ws == None:
            ws = pyarts.workspace.Workspace()
            self.new_ws = True
        self.ws = ws


    def calculate(self, load_if_exist=False, recalculate=False):
        if self.check_existing_lut():
            if not recalculate:
                print("The Lookup Table is already calculated.")
                if load_if_exist:
                    self.load()
                return
            print('The Lookuptable will be recalculated.')

        if self.new_ws:
            print('Necessary quantities are loaded.')
            self.lut_setup()
        print('The Lookup Table calculation is starting.')
        with ty.utils.Timer():
            self.calculate_lut()
        print('Finished with Lookup Table calculation.')


    def lut_setup(self):
        self.ws.LegacyContinuaInit()
        self.ws.PlanetSet(option="Earth")        

        self.f_grid_from_spectral_grid()
        self.ws.stokes_dim = 1
        self.ws.AtmosphereSet1D()
        self.ws.batch_atm_fields_compact = pyarts.xml.load(f'{self.exp_setup.rfmip_path}{self.exp_setup.input_folder}atm_fields.xml')
        species = pyarts.xml.load(f"{self.exp_setup.rfmip_path}{self.exp_setup.input_folder}species.xml")
        self.add_species(species)


    def calculate_lut(self):
        if not os.path.exists(f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/'):
            os.mkdir(f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/')

        # Read a line file and a matching small frequency grid
        self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
        basename=f'{self.exp_setup.arts_data_path}arts-cat-data/lines/'
        )

        self.ws.ReadXsecData(
            basename=f'{self.exp_setup.arts_data_path}arts-cat-data/xsec/'
        )

        self.ws.abs_lines_per_speciesCutoff(option='ByLine', value=750e9)
        self.ws.abs_lines_per_speciesCompact()

        self.ws.propmat_clearsky_agendaAuto()

        self.ws.abs_lookupSetupBatch()
        self.ws.lbl_checkedCalc()
        self.ws.abs_lookupCalc()

        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        self.ws.WriteXML('binary', self.ws.abs_lookup, f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/lookup.xml')


    def load(self):
        """ Loads existing Lookup table and adjust it for the calculation. """
        self.ws.propmat_clearsky_agendaAuto()

        self.ws.ReadXML(self.ws.abs_lookup, f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/lookup.xml')

        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        if not self.new_ws:
            self.ws.abs_lookupAdapt() # Adapts a gas absorption lookup table to the current calculation. Removes unnessesary freqs are species.


    def f_grid_from_spectral_grid(self):
        if self.exp_setup.which_spectral_grid == 'frequency':
            f_grid = np.linspace(self.exp_setup.spectral_grid['min'], self.exp_setup.spectral_grid['max'], self.exp_setup.spectral_grid['n'], endpoint=True)
        elif self.exp_setup.which_spectral_grid == 'wavelength':
            lam_grid = np.linspace(self.exp_setup.spectral_grid['min'], self.exp_setup.spectral_grid['max'], self.exp_setup.spectral_grid['n'], endpoint=True)*1e-9
            f_grid = ty.physics.wavelength2frequency(lam_grid)[::-1]
        elif self.exp_setup.which_spectral_grid == 'kayser':
            kayser_grid = np.linspace(self.exp_setup.spectral_grid['min'], self.exp_setup.spectral_grid['max'], self.exp_setup.spectral_grid['n'], endpoint=True)
            f_grid = ty.physics.wavenumber2frequency(kayser_grid)

        self.ws.f_grid = f_grid


    def add_species(self, species):
        # adding Line MIxing and Self Continuum for some species
        if 'abs_species-H2O' in species:
            replace_values(species, 'abs_species-H2O', ['abs_species-H2O', 'abs_species-H2O-SelfContCKDMT252', 'abs_species-H2OForeignContCKDMT252'])
        if 'abs_species-CO2' in species:
            replace_values(species, 'abs_species-CO2', ['abs_species-CO2', 'abs_species-CO2-LM', 'abs_species-CO2-CKDMT252'])
        if 'abs_species-O3' in species:
            replace_values(species, 'abs_species-O3', ['abs_species-O3', 'abs_species-O3-XFIT'])
        if 'abs_species-O2' in species:
            replace_values(species, 'abs_species-O2', ['abs_species-O2', 'abs_species-O2-CIAfunCKDMT100'])
        if 'abs_species-N2' in species:
            replace_values(species, 'abs_species-N2', ['abs_species-N2', 'abs_species-N2-CIAfunCKDMT252', 'abs_species-N2-CIAfunCKDMT252'])

        species = [spec[12:] for spec in species]

        self.ws.abs_speciesSet(species=species)


    def check_existing_lut(self):
        return os.path.exists(f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/lookup.xml')


def replace_values(list_to_replace, item_to_replace, item_to_replace_with):
    return [item_to_replace_with if item == item_to_replace else item for item in list_to_replace]


def main():
    exp = ExperimentSetup(
        name='lut',
        description='testing lookup table',
        rfmip_path='/Users/jpetersen/rare/rfmip/',
        input_folder='input/rfmip/',
        arts_data_path='/Users/jpetersen/rare/',
        solar_type='None',
        which_spectral_grid='wavelength',
        spectral_grid={'min': 380, 'max': 780, 'n': 300},
        species=['water_vapor'],
        angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
    )
    
    with ty.utils.Timer():
        lut = BatchLookUpTable(exp)
        lut.calculate(recalculate=True)
    

if __name__ == '__main__':
    main()
