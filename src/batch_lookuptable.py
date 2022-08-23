import os
import pyarts
import numpy as np
import typhon as ty
from itertools import repeat
from multiprocessing import Pool

import write_xml_input_data as input_data
from experiment_setup import ExperimentSetup


def calc_lookup_parallel(exp_setup, n_procs=8, recalculate=False):

    if os.path.exists(f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/lookup.xml'):
        print("The Lookup Table already exists.")
        if not recalculate:
            return

    with Pool(n_procs) as p:
        p.starmap(calc_lookup, list(zip(repeat(exp_setup, n_procs), repeat(n_procs, n_procs), list(range(n_procs)), repeat(recalculate))))

    # should only be used if the same species are in the luts!!
    combine_luts(exp_setup, n_procs=n_procs)


def combine_luts(exp_setup, n_procs=8):
    lut_list = [
        pyarts.xml.load(
            f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/lookup_{str(i).zfill(2)}.xml'
        ) for i in range(n_procs)
    ]
    main_lut = lut_list[0]
    for lut in lut_list[1:]:
        main_lut.f_grid = pyarts.arts.Vector(np.concatenate((main_lut.f_grid, lut.f_grid), axis=0))
        main_lut.xsec = pyarts.arts.Tensor4(np.concatenate((main_lut.xsec, lut.xsec), axis=2))

    pyarts.xml.save(main_lut, f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/lookup.xml')


def calc_lookup(exp_setup, n_procs, proc_id, recalculate=False):
    lut = LookUpTable(exp_setup, n_slice=n_procs, active_slice=proc_id)
    lut.calculate(recalculate=recalculate)

class LookUpTable():
    def __init__(self, exp_setup, ws=None, n_slice=None, active_slice=1):
        self.active_slice = active_slice
        self.n_slice = n_slice
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
        self.ws.execute_controlfile("general/agendas.arts")
        self.ws.execute_controlfile("general/planet_earth.arts")

        self.f_grid_from_spectral_grid()
        self.ws.stokes_dim = 1
        self.ws.AtmosphereSet1D()
        self.ws.batch_atm_fields_compact = pyarts.xml.load(f'{self.exp_setup.rfmip_path}{self.exp_setup.input_folder}atm_fields.xml')
        species = pyarts.xml.load(f"{self.exp_setup.rfmip_path}{self.exp_setup.input_folder}species.xml")
        self.add_species(species)

        # Read a line file and a matching small frequency grid
        self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
        basename=f'{self.exp_setup.arts_data_path}arts-cat-data/lines/'
        )

        self.ws.ReadXsecData(
            basename=f'{self.exp_setup.arts_data_path}arts-cat-data/xsec/'
        )

        self.ws.abs_lines_per_speciesCutoff(option='ByLine', value=750e9)
        self.ws.abs_lines_per_speciesCompact()


    def calculate_lut(self, proc_id=None, n_proc=8):
        if not os.path.exists(f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/'):
            os.mkdir(f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/')

        self.ws.propmat_clearsky_agendaAuto()

        self.ws.abs_lookupSetupBatch()
        self.ws.lbl_checkedCalc()
        self.ws.abs_lookupCalc()

        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        if self.n_slice == None:
            self.ws.WriteXML('binary', self.ws.abs_lookup, f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/lookup.xml')
        else:
            self.ws.WriteXML('binary', self.ws.abs_lookup, f'{self.exp_setup.rfmip_path}lookup_tables/{self.exp_setup.name}/lookup_{str(self.active_slice).zfill(2)}.xml')


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

        if self.n_slice is not None:
            self.ws.f_grid = get_chunk(f_grid, self.n_slice, self.active_slice)
        else:
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


def get_chunk(lst, n_chunks, chunk_id):
    slice_len = len(lst)//n_chunks
    return lst[slice_len*chunk_id:slice_len*(chunk_id+1)]


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
    # input_data.create_input_data(exp_setup=exp)
    with ty.utils.Timer():
        calc_lookup_parallel(exp, 2)
    
    with ty.utils.Timer():
        lut = LookUpTable(exp)
        lut.calculate(recalculate=True)
    

if __name__ == '__main__':
    main()