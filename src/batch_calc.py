import os
import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from experiment_setup import read_exp_setup
from batch_lookuptable import BatchLookUpTable

def run_arts_batch(exp_setup, verbosity=3):
    """Run Arts Calculation for RFMIP. """

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    
    ws.LegacyContinuaInit()
    ws.PlanetSet(option="Earth")

    ws.IndexCreate('planck_emission')

    ws.NumericCreate('z0')
    ws.NumericCreate('surface_reflectivity_numeric')

    ws.VectorCreate('surface_temperatures')
    ws.VectorCreate('surface_reflectivities')
    ws.VectorCreate('surface_altitudes')

    # Number of Stokes components to be computed
    print('setup and reading')
    ws.stokes_dim = 1

    # No jacobian calculation
    ws.jacobianOff()

    # Frequency grid
    if exp_setup.which_spectral_grid == 'frequency':  
        ws.f_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)
    elif exp_setup.which_spectral_grid == 'wavelength':
        lam_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)*1e-9
        ws.f_grid = ty.physics.wavelength2frequency(lam_grid)[::-1]
    elif exp_setup.which_spectral_grid == 'kayser':
        kayser_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)*1e2
        ws.f_grid = ty.physics.wavenumber2frequency(kayser_grid)
    else:
        print('Use a valid option fo which grid to use. Option are frequency, wavelength or kayser.')

    # set geographical position
    # always at 0/0 because the atm is 1D anyways but the sun zenioth angle varies
    ws.lat_true = [0.0]
    ws.lon_true = [0.0]

    # No sensor properties
    ws.sensorOff()
    
    # Definition of sensor position and line of sight (LOS)
    ws.sensor_pos = [[0.0]] # flux are calculated at every ppath point
    ws.sensor_los = [[0]] # irrelevant for fluxes

    ## Atmosphere
    ws.AtmosphereSet1D()
    ws.ArrayOfVectorCreate('p_grids')
    ws.p_grids = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}pressure_layer.xml')
    ws.batch_atm_fields_compact = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}atm_fields.xml')

    species = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}species.xml")
    add_species(ws, species)

    ## Lookup Table
    lut = BatchLookUpTable(exp_setup=exp_setup, ws=ws)
    lut.calculate(load_if_exist=True, optimise_speed=True)

    ## Surface
    # set surface resolution
    ws.MatrixSetConstant(ws.z_surface, 1, 1, 0)

    ws.surface_temperatures = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}surface_temperature.xml')
    ws.surface_reflectivities = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}surface_albedo.xml')
    ws.surface_altitudes = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}surface_altitudes.xml')
    
  
    # Star or no star settings
    if exp_setup.solar_type == 'None':
        ws.gas_scattering_do = 0
        ws.dobatch_calc_agenda = dobatch_calc_agenda__disort
        ws.gas_scattering_agenda = gas_scattering_agenda__Rayleigh

    else:
        ws.gas_scattering_do = 1
        ws.gas_scattering_agenda = gas_scattering_agenda__Rayleigh
        ws.NumericCreate('solar_zenith_angle')
        ws.VectorCreate('solar_zenith_angles')
        ws.solar_zenith_angles = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}solar_zenith_angle.xml')
    
        if exp_setup.solar_type == 'BlackBody':
            ws.dobatch_calc_agenda = dobatch_calc_agenda__disort_blackbody
        elif exp_setup.solar_type == 'Spectrum':
            ws.ArrayOfGriddedField2Create('star_spectras')
            ws.GriddedField2Create('star_spectrum_raw')
            ws.star_spectras = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}star_spectra.xml')
            ws.dobatch_calc_agenda = dobatch_calc_agenda__disort_spectrum


    # set planck emission
    ws.planck_emission = int(exp_setup.planck_emission)

    # set angular grid
    ws.AngularGridsSetFluxCalc(N_za_grid=exp_setup.angular_grid['N_za_grid'], N_aa_grid=exp_setup.angular_grid['N_aa_grid'], za_grid_type=exp_setup.angular_grid['za_grid_type'])
    ws.aa_grid.value += 180. # disort goes from 0 t0 360 

    ws.sensor_checkedCalc()

    ws.ybatch_start = 0
    ws.ybatch_n = len(ws.surface_temperatures.value) # loop over all sites

    with ty.utils.Timer():
        print('starting calculation')
        ws.DOBatchCalc()

    if not os.path.exists(f'{exp_setup.rfmip_path}output/{exp_setup.name}/'):
        os.mkdir(f'{exp_setup.rfmip_path}output/{exp_setup.name}/')
    
    print('saving')
    ws.WriteXML('binary', ws.dobatch_spectral_irradiance_field, f'{exp_setup.rfmip_path}output/{exp_setup.name}/spectral_irradiance.xml')


@pyarts.workspace.arts_agenda(allow_callbacks=False)
def dobatch_calc_agenda__disort(ws):
    # print batch
    ws.Print(ws.ybatch_index, 0)

    # set new atmosphere for batch
    ws.Extract(ws.p_grid, ws.p_grids, ws.ybatch_index) # set p_grid for batch
    ws.Extract(ws.atm_fields_compact, ws.batch_atm_fields_compact, ws.ybatch_index) # sets new atm
    ws.Extract(ws.surface_skin_t, ws.surface_temperatures, ws.ybatch_index)
    ws.MatrixSetConstant(ws.t_surface, 1, 1, ws.surface_skin_t)
    ws.Extract(ws.surface_reflectivity_numeric, ws.surface_reflectivities, ws.ybatch_index)
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, ws.surface_reflectivity_numeric)
    ws.Extract(ws.z0, ws.surface_altitudes, ws.ybatch_index)
    ws.MatrixSetConstant(ws.z_surface, 1, 1, ws.z0)

    # recalcs the atmosphere
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # set partical scattering
    ws.cloudboxSetFullAtm()
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()

    # Checks 
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.scat_data_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    # Calculation
    ws.DisortCalcIrradiance(nstreams=10, quiet=0, emission=ws.planck_emission)

    # free fields
    ws.Touch(ws.spectral_radiance_field)
    ws.Touch(ws.radiance_field)
    ws.Touch(ws.cloudbox_field)
    ws.Touch(ws.irradiance_field)


@pyarts.workspace.arts_agenda(allow_callbacks=False)
def dobatch_calc_agenda__disort_blackbody(ws):
    # print batch
    ws.Print(ws.ybatch_index, 0)

    # set new atmosphere for batch
    ws.Extract(ws.p_grid, ws.p_grids, ws.ybatch_index) # set p_grid for batch
    ws.Extract(ws.atm_fields_compact, ws.batch_atm_fields_compact, ws.ybatch_index) # sets new atm
    ws.Extract(ws.surface_skin_t, ws.surface_temperatures, ws.ybatch_index)
    ws.MatrixSetConstant(ws.t_surface, 1, 1, ws.surface_skin_t)
    ws.Extract(ws.surface_reflectivity_numeric, ws.surface_reflectivities, ws.ybatch_index)
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, ws.surface_reflectivity_numeric)
    ws.Extract(ws.z0, ws.surface_altitudes, ws.ybatch_index)
    ws.MatrixSetConstant(ws.z_surface, 1, 1, ws.z0)
    ws.Extract(ws.solar_zenith_angle, ws.solar_zenith_angles, ws.ybatch_index)

    ws.starsAddSingleBlackbody(longitude=ws.solar_zenith_angle)

    # recalcs the atmosphere
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # set partical scattering
    ws.cloudboxSetFullAtm()
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()

    # Checks 
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.scat_data_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    # Calculation
    ws.DisortCalcIrradiance(nstreams=10, quiet=0, emission=ws.planck_emission)

    # free fields
    ws.Touch(ws.spectral_radiance_field)
    ws.Touch(ws.radiance_field)
    ws.Touch(ws.cloudbox_field)
    ws.Touch(ws.irradiance_field)


@pyarts.workspace.arts_agenda(allow_callbacks=False)
def dobatch_calc_agenda__disort_spectrum(ws):
    # print batch
    ws.Print(ws.ybatch_index, 0)

    # set new atmosphere for batch
    ws.Extract(ws.p_grid, ws.p_grids, ws.ybatch_index) # set p_grid for batch
    ws.Extract(ws.atm_fields_compact, ws.batch_atm_fields_compact, ws.ybatch_index) # sets new atm
    ws.Extract(ws.surface_skin_t, ws.surface_temperatures, ws.ybatch_index)
    ws.MatrixSetConstant(ws.t_surface, 1, 1, ws.surface_skin_t)
    ws.Extract(ws.surface_reflectivity_numeric, ws.surface_reflectivities, ws.ybatch_index)
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, ws.surface_reflectivity_numeric)
    ws.Extract(ws.z0, ws.surface_altitudes, ws.ybatch_index)
    ws.MatrixSetConstant(ws.z_surface, 1, 1, ws.z0)
    ws.Extract(ws.solar_zenith_angle, ws.solar_zenith_angles, ws.ybatch_index)
    ws.Extract(ws.star_spectrum_raw, ws.star_spectras, ws.ybatch_index)

    ws.starsAddSingleFromGrid(star_spectrum_raw=ws.star_spectrum_raw, temperature=0, longitude=ws.solar_zenith_angle)

    # recalcs the atmosphere
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # set partical scattering
    ws.cloudboxSetFullAtm()
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()

    # Checks 
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.scat_data_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    # Calculation
    ws.DisortCalcIrradiance(nstreams=10, quiet=0, emission=ws.planck_emission)

    # free fields
    ws.Touch(ws.spectral_radiance_field)
    ws.Touch(ws.radiance_field)
    ws.Touch(ws.cloudbox_field)
    ws.Touch(ws.irradiance_field)
    

#gas scattering agenda
@pyarts.workspace.arts_agenda
def gas_scattering_agenda__Rayleigh(ws):
    ws.Ignore(ws.rtp_vmr)
    ws.gas_scattering_coefAirSimple()
    ws.gas_scattering_matRayleigh()


def add_species(ws, species):
    # NO NH3!!
    # replace CO2 with CO2 LineMixing
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
    
    ws.abs_speciesSet(species=species)


def replace_values(list_to_replace, item_to_replace, item_to_replace_with):
    return [item_to_replace_with if item == item_to_replace else item for item in list_to_replace]


def main():
    exp = read_exp_setup(exp_name='test', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    print(exp)
    run_arts_batch(exp)


if __name__ == '__main__':
    main()
