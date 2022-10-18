import os
import sys
import pyarts
import argparse
import numpy as np
import typhon as ty

import write_xml_input_data as input_data
from experiment_setup import ExperimentSetup, read_exp_setup

class BatchLookUpTable():
    def __init__(self, exp_setup, ws=None, n_chunks: int = 0):
        self.exp_setup = exp_setup
        self.new_ws = False
        if ws == None:
            ws = pyarts.workspace.Workspace()
            self.new_ws = True
        self.ws = ws
        if n_chunks == 0 or n_chunks == 1:
            self.n_chunks=None
        elif 2 <= n_chunks <= 99: 
            self.n_chunks=n_chunks
        else:
            raise ValueError(f'`n_chunks` must be between 2 and 99 (or 0 / 1 for no splitting) but is {n_chunks}!')


    def calculate(self, load_if_exist=False, recalculate=False, optimise_speed=False, chunk_id=None):
        self.chunk_id=chunk_id
        if self.check_existing_lut():
            if not recalculate:
                print("The Lookup Table is already calculated.")
                if load_if_exist:
                    self.load(optimise_speed=optimise_speed)
                return
            print('The Lookuptable will be recalculated.')

        if self.new_ws:
            print('Necessary quantities are loaded.')
            self.lut_setup()
        print('The Lookup Table calculation is starting.')
        with ty.utils.Timer():
            self.calculate_lut(optimise_speed=optimise_speed)
        print('Finished with Lookup Table calculation.')


    def lut_setup(self):
        # self.ws.LegacyContinuaInit()
        self.ws.PlanetSet(option="Earth")        

        self.f_grid_from_spectral_grid()
        self.ws.stokes_dim = 1
        self.ws.AtmosphereSet1D()
        self.ws.batch_atm_fields_compact = pyarts.xml.load(f'{self.exp_setup.rfmip_path}{self.exp_setup.input_folder}atm_fields.xml')
        species = pyarts.xml.load(f"{self.exp_setup.rfmip_path}{self.exp_setup.input_folder}species.xml")
        self.add_species(species)
        

    def calculate_lut(self, optimise_speed=False):
        # Read a line file and a matching small frequency grid
        self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
        basename=f'{self.exp_setup.arts_data_path}arts-cat-data/lines/'
        )

        self.ws.ReadXsecData(
            basename=f'{self.exp_setup.arts_data_path}arts-cat-data/xsec/'
        )

        if optimise_speed:
            self.ws.abs_lines_per_speciesCutoff(option='ByLine', value=750e9)
            self.ws.abs_lines_per_speciesCompact()

        self.ws.propmat_clearsky_agendaAuto()

        self.ws.abs_lookupSetupBatch()
        self.ws.lbl_checkedCalc()
        self.ws.abs_lookupCalc()

        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
        
        savename = (
            f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.lookuptable}'
            if self.n_chunks is None
            else f'{self.exp_setup.rfmip_path}lookup_tables/chunk{str(self.chunk_id).zfill(2)}_{self.exp_setup.lookuptable}'
        )
        self.ws.abs_lookup.value.savexml(file=savename, type='binary')


    def load(self, optimise_speed=False):
        """ Loads existing Lookup table and adjust it for the calculation. """
        self.ws.Touch(self.ws.abs_lines_per_species)
        if optimise_speed:
            self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
            self.ws.abs_lines_per_speciesCompact()

        self.ws.propmat_clearsky_agendaAuto()

        self.ws.ReadXML(self.ws.abs_lookup, f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.lookuptable}')

        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        if not self.new_ws:
            self.ws.abs_lookupAdapt() # Adapts a gas absorption lookup table to the current calculation. Removes unnessesary freqs are species.

        self.ws.lbl_checked = 1


    def f_grid_from_spectral_grid(self):
        if self.exp_setup.which_spectral_grid == 'frequency':
            f_grid = np.linspace(self.exp_setup.spectral_grid['min'], self.exp_setup.spectral_grid['max'], self.exp_setup.spectral_grid['n'], endpoint=True)
        elif self.exp_setup.which_spectral_grid == 'wavelength':
            lam_grid = np.linspace(self.exp_setup.spectral_grid['min'], self.exp_setup.spectral_grid['max'], self.exp_setup.spectral_grid['n'], endpoint=True)*1e-9
            f_grid = ty.physics.wavelength2frequency(lam_grid)[::-1]
        elif self.exp_setup.which_spectral_grid == 'kayser':
            kayser_grid = np.linspace(self.exp_setup.spectral_grid['min'], self.exp_setup.spectral_grid['max'], self.exp_setup.spectral_grid['n'], endpoint=True)*1e2
            f_grid = ty.physics.wavenumber2frequency(kayser_grid)

        if self.n_chunks is not None:
            f_grid = get_chunk(f_grid, self.n_chunks, self.chunk_id)
        self.ws.f_grid = f_grid

       
    def add_species(self, species):
        if "abs_species-O3" in species:
            species.append("abs_species-O3-XFIT")
        if 'abs_species-H2O' in species:
            species = replace_values(species, 'abs_species-H2O', 'abs_species-H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350')
        if 'abs_species-CO2' in species:
            species = replace_values(species, 'abs_species-CO2', 'abs_species-CO2, CO2-CKDMT252')
        if 'abs_species-O2' in species:
            species = replace_values(species, 'abs_species-O2', 'abs_species-O2, O2-CIAfunCKDMT100')
        if 'abs_species-N2' in species:
            species = replace_values(species, 'abs_species-N2', 'abs_species-N2, N2-CIAfunCKDMT252, N2-CIAfunCKDMT252')
        
        species = [spec[12:] for spec in species]
        self.ws.abs_speciesSet(species=species)


    def check_existing_lut(self):
        return os.path.exists(f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.lookuptable}')


def replace_values(list_to_replace, item_to_replace, item_to_replace_with):
    return [item_to_replace_with if item == item_to_replace else item for item in list_to_replace]


def get_chunk(arr, n_chunks, chunk_id):
    slice_len = np.shape(arr)[0]//n_chunks
    return arr[slice_len*chunk_id:slice_len*(chunk_id+1)]

 
def combine_luts(exp_setup, n_chunks=8):
    if exp_setup is None:
        exp_setup = ExperimentSetup(
            name='lut',
            description='testing lookup table',
            rfmip_path='/Users/jpetersen/rare/rfmip/',
            input_folder='input/rfmip/',
            arts_data_path='/Users/jpetersen/rare/',
            lookuptable='lut_test.xml',
            solar_type='None',
            planck_emission='0',
            which_spectral_grid='wavelength',
            spectral_grid={'min': 380, 'max': 780, 'n': 10},
            species=['water_vapor'],
            angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'}
        )
    print('Combining luts')
    lut_list = [
        pyarts.xml.load(
            f'{exp_setup.rfmip_path}lookup_tables/chunk{str(i).zfill(2)}_{exp_setup.lookuptable}'
        ) for i in range(n_chunks)
    ]
    main_lut = lut_list[0]
    for lut in lut_list[1:]:
        main_lut.f_grid = pyarts.arts.Vector(np.concatenate((main_lut.f_grid, lut.f_grid), axis=0))
        main_lut.xsec = pyarts.arts.Tensor4(np.concatenate((main_lut.xsec, lut.xsec), axis=2))

    main_lut.savexml(file=f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.lookuptable}', type='binary')


def main(exp=None, n_chunks=0, chunks_id=None):
    if exp is None:
        exp = ExperimentSetup(
            name='lut',
            description='testing lookup table',
            rfmip_path='/Users/jpetersen/rare/rfmip/',
            input_folder='input/rfmip/',
            arts_data_path='/Users/jpetersen/rare/',
            lookuptable='lut_test.xml',
            solar_type='None',
            planck_emission='0',
            which_spectral_grid='wavelength',
            spectral_grid={'min': 380, 'max': 780, 'n': 10},
            species=['water_vapor'],
            angular_grid={'N_za_grid': 20, 'N_aa_grid': 41, 'za_grid_type': 'linear_mu'},
            h2o="H2O, H2O-SelfContCKDMT320, H2O-ForeignContCKDMT320"
        )
    
    with ty.utils.Timer():
        lut = BatchLookUpTable(exp, n_chunks=n_chunks)
        lut.calculate(recalculate=True, chunk_id=chunks_id, optimise_speed=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguent parser for lookuptable calculation",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--experiment_setups", type=str, nargs='*', help="Experiment_setups for which the lookuptabe will be calculated", required=False, default=[None])                               
    parser.add_argument("-c", "--chunks", type=int, nargs=2, help="Dived lut calculation in differend f_grid in NCHUNKS chunks and uses CHUNK_ID for the current calculation.", metavar='NCHUNKS CHUNK_ID', required=False, default=[0, None])
    parser.add_argument("--combine_luts", action="store_true", help="Flage to combine the lookuptables")
    config = vars(parser.parse_args())
    print(config)
    
    if config['combine_luts']:
        for exp_name in config['experiment_setups']:
            exp = exp_name
            if exp_name is not None:
                exp_setup_path = f'{os.getcwd()}/experiment_setups/'
                exp = read_exp_setup(exp_name=str(exp_name), path=exp_setup_path)
            combine_luts(exp, n_chunks=config['chunks'][0])
        exit()

    for exp_name in config['experiment_setups']:
        exp = exp_name
        if exp_name is not None:
            exp_setup_path = f'{os.getcwd()}/experiment_setups/'
            exp = read_exp_setup(exp_name=str(exp_name), path=exp_setup_path)
            if not os.path.exists(f'{exp.rfmip_path}{exp.input_folder}'):
                input_data.create_input_data(exp)
        main(exp=exp, n_chunks=config['chunks'][0], chunks_id=config['chunks'][1])
