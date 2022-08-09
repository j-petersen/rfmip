import os
import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from experiment_setup import read_exp_setup
from lookuptable import LookUpTable

def run_arts_batch(exp_setup, verbosity=3):
    """Run Arts Calculation for RFMIP. """
    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")

    ws.NumericCreate('z0')
    ws.NumericCreate('surface_reflectivity_numeric')

    ws.VectorCreate('surface_temperatures')
    ws.VectorCreate('surface_reflectivities')
    ws.VectorCreate('surface_altitudes')

    ## Set Agendas 
    # cosmic background radiation
    ws.iy_space_agenda = ws.iy_space_agenda__CosmicBackground

    ws.surface_rtprop_agenda = ws.surface_rtprop_agenda__lambertian_ReflFix_SurfTFromt_field

    # sensor-only path
    ws.ppath_agenda = ws.ppath_agenda__FollowSensorLosPath

    # Geometric Ppath (no refraction)
    ws.ppath_step_agenda = ws.ppath_step_agenda__GeometricPath

    ws.iy_surface_agenda = iy_surface_agenda # egal für disort?
    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    # ws.iy_surface_agenda = ws.iy_surface_agenda__UseSurfaceRtprop

    # Number of Stokes components to be computed
    print('setup and reading')
    ws.stokes_dim = 1

    # Reference ellipsoid
    ws.refellipsoidEarth(ws.refellipsoid, "Sphere")

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

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
    ws = add_species(ws, species)

    # Read a line file and a matching small frequency grid
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
       basename=f'{exp_setup.arts_data_path}arts-cat-data/lines/'
    )

    ws.ReadXsecData(
        basename=f'{exp_setup.arts_data_path}arts-cat-data/xsec/'
    )

    ## Lookup Table
    lut = LookUpTable(exp_setup=exp_setup, ws=ws)
    lut.calculate(load_if_exist=True)

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
    else:
        ws.gas_scattering_do = 1
        ws.gas_scattering_agenda = gas_scattering_agenda
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

    # set angular grid
    ws.AngularGridsSetFluxCalc(N_za_grid=exp_setup.angular_grid['N_za_grid'], N_aa_grid=exp_setup.angular_grid['N_aa_grid'], za_grid_type=exp_setup.angular_grid['za_grid_type'])
    ws.aa_grid.value += 180. # disort goes from 0 t0 360 

    # Check model atmosphere
    ws.scat_data_checkedCalc()
    ws.lbl_checkedCalc()
    ws.sensor_checkedCalc()

    ws.ybatch_start = 0
    ws.ybatch_n = len(ws.surface_temperatures.value) # loop over all sites

    with ty.utils.Timer():
        print('starting calculation')
        ws.DOBatchCalc()

    # irrad = np.squeeze(ws.dobatch_spectral_irradiance_field.value)
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

    # Checks 
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    # Calculation
    ws.DisortCalcClearSky(nstreams=6, quiet=0)
    ws.spectral_irradiance_fieldFromSpectralRadianceField()

    # free fields
    ws.Tensor5SetConstant(ws.radiance_field, 0, 0, 0, 0, 0, 0.)
    ws.Tensor4SetConstant(ws.irradiance_field, 0, 0, 0, 0, 0.)
    ws.Tensor7SetConstant(ws.cloudbox_field, 0, 0, 0, 0, 0, 0, 0, 0.)


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

    # Checks 
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    # Calculation
    ws.DisortCalcClearSky(nstreams=6, quiet=0)
    ws.spectral_irradiance_fieldFromSpectralRadianceField()

    # free fields
    ws.Tensor5SetConstant(ws.radiance_field, 0, 0, 0, 0, 0, 0.)
    ws.Tensor4SetConstant(ws.irradiance_field, 0, 0, 0, 0, 0.)
    ws.Tensor7SetConstant(ws.cloudbox_field, 0, 0, 0, 0, 0, 0, 0, 0.)


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

    # Checks 
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    # Calculation
    ws.DisortCalcClearSky(nstreams=6, quiet=0)
    ws.spectral_irradiance_fieldFromSpectralRadianceField()

    # free fields
    ws.Tensor5SetConstant(ws.radiance_field, 0, 0, 0, 0, 0, 0.)
    ws.Tensor4SetConstant(ws.irradiance_field, 0, 0, 0, 0, 0.)
    ws.Tensor7SetConstant(ws.cloudbox_field, 0, 0, 0, 0, 0, 0, 0, 0.)
    

#gas scattering agenda
@pyarts.workspace.arts_agenda
def gas_scattering_agenda(ws):
    ws.Ignore(ws.rtp_vmr)
    ws.gas_scattering_coefAirSimple()
    ws.gas_scattering_matRayleigh()

#surface scattering agenda
@pyarts.workspace.arts_agenda
def iy_surface_agenda(ws):

    ws.Ignore(ws.iy_transmittance)
    ws.Ignore(ws.iy_id)
    ws.Ignore(ws.iy_main_agenda)
    ws.Ignore(ws.rtp_los)
    ws.Ignore(ws.rte_pos2)
    ws.Ignore(ws.diy_dx)
    ws.Touch(ws.diy_dx)

    ws.iySurfaceInit()
    ws.Ignore(ws.dsurface_rmatrix_dx)
    ws.Ignore(ws.dsurface_emission_dx)
    ws.Ignore(ws.surface_props_data)
    ws.Ignore(ws.dsurface_names)

    # ws.iySurfaceLambertian()
    ws.iySurfaceLambertianDirect()


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
    return ws


def replace_values(list_to_replace, item_to_replace, item_to_replace_with):
    return [item_to_replace_with if item == item_to_replace else item for item in list_to_replace]


def main():
    exp = read_exp_setup(exp_name='solar_angle', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    print(exp)
    run_arts_batch(exp)


if __name__ == '__main__':
    main()