import os
import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt

from experiment_setup import read_exp_setup

def run_arts_batch(exp_setup, verbosity=2):
    """Run Arts Calculation for RFMIP. """
    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")

    ws.NumericCreate('z0')
    ws.NumericCreate('surface_reflectivity_numeric')

    ws.VectorCreate('surface_temperatures')
    ws.VectorCreate('surface_reflectivities')
    ws.VectorCreate('surface_altitudes')

    ## Set Agendas 
    # Agenda for scalar gas absorption calculation
    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    # cosmic background radiation
    ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground)

    ws.Copy(ws.surface_rtprop_agenda, ws.surface_rtprop_agenda__lambertian_ReflFix_SurfTFromt_field)

    # sensor-only path
    ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)

    # Geometric Ppath (no refraction)
    ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)

    ws.iy_surface_agenda = iy_surface_agenda # egal für disort?
    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    # ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)

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
        kayser_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)
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

    ws.abs_lines_per_speciesSetCutoff(option="ByLine", value=750e9)

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Set propmat agenda
    ws.propmat_clearsky_agendaSetAutomatic()

    ## Lookup Table
    ws = lookup_table(ws, exp_setup)

    ## Surface
    # set surface resolution
    ws.MatrixSetConstant(ws.z_surface, 1, 1, 0)

    
    ws.surface_temperatures = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}surface_temperature.xml')
    ws.surface_reflectivities = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}surface_albedo.xml')
    ws.surface_altitudes = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}surface_altitudes.xml')
    ws.solar_zenith_angles = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}solar_zenith_angle.xml')
    
    #Switch on gas scattering
    ws.gas_scattering_do = exp_setup.gas_scattering_do
    if exp_setup.gas_scattering_do:
            ws.gas_scattering_agenda = gas_scattering_agenda
  
    # Star or no star settings
    if exp_setup.solar_type == 'None':
        ws.gas_scattering_do = 0
    else:
        # add star
        ws.gas_scattering_do = 1
        ws.gas_scattering_agenda = gas_scattering_agenda
        ws.NumericCreate('solar_zenith_angle')
        ws.VectorCreate('solar_zenith_angles')
        if exp_setup.solar_type == 'Spectrum':
            ws.ArrayOfGriddedField2Create('star_spectras')
            ws.star_spectra = pyarts.xml.load(f'{exp_setup.rfmip_path}{exp_setup.input_folder}star_spectrum.xml')

    # set angular grid
    ws.AngularGridsSetFluxCalc(N_za_grid=exp_setup.angular_grid['N_za_grid'], N_aa_grid=exp_setup.angular_grid['N_aa_grid'], za_grid_type=exp_setup.angular_grid['za_grid_type'])
    ws.aa_grid.value += 180. # disort goes from 0 t0 360 

    # Check model atmosphere
    ws.scat_data_checkedCalc()
    ws.abs_xsec_agenda_checkedCalc()
    ws.lbl_checkedCalc()
    ws.sensor_checkedCalc()

    ws.ybatch_start = 0
    ws.ybatch_n = len(ws.surface_temperatures.value) # loop over all sites

    # change batchcalc agenda based on star type
    if exp_setup.solar_type == 'None':
        ws.dobatch_calc_agenda = dobatch_calc_agenda__disort
    elif exp_setup.solar_type == 'BlackBody':
        ws.dobatch_calc_agenda = dobatch_calc_agenda__disort_blackbody
    elif exp_setup.solar_type == 'Spectrum':
        ws.dobatch_calc_agenda = dobatch_calc_agenda__disort_spectrum

        print('Not yet implimented!')

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

    ws.starBlackbodySimple(distance=1.5e11, latitude=0, longitude=ws.solar_zenith_angle)

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

    ws.starFromGrid(temperature=0, distance=1.5e11, latitude=0, longitude=ws.solar_zenith_angle)

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
    ws.gas_scatteringCoefAirSimple()
    ws.gas_scatteringMatrixRayleigh()

#Main agenda
@pyarts.workspace.arts_agenda
def iy_main_agenda_ClearSky(ws):
    ws.ppathCalc()
    ws.iyClearsky()

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

def lookup_table(ws, exp_setup):
    if os.path.exists(f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/lookup_table'):
        # load LUT
        ws.ReadXML(ws.abs_lookup, f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/lookup.xml')
        ws.abs_lookupAdapt() # Adapts a gas absorption lookup table to the current calculation. Removes unnessesary freqs are species.
    else:
        # re calc and save LUT
        if not os.path.exists(f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/'):
            os.mkdir(f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/')

        ws.abs_lookupSetupBatch()
        ws.abs_xsec_agenda_checkedCalc()
        ws.lbl_checkedCalc()
        ws.abs_lookupCalc()

        ws.WriteXML('binary', ws.abs_lookup, f'{exp_setup.rfmip_path}lookup_tables/{exp_setup.name}/lookup.xml')
    return ws

def replace_item_in_list(item_list, old_val, new_val):
    return list(map(lambda x: x.replace(old_val, new_val), item_list))

def replace_values(list_to_replace, item_to_replace, item_to_replace_with):
    return [item_to_replace_with if item == item_to_replace else item for item in list_to_replace]

def solar_spectrum(ws, solar_zenith_angle=0.0, star_type=None):
    ## set star
    ws.Touch(ws.stars)
    
    if star_type==None:
        ws.star_do = 0
    elif star_type=='BlackBody':
        ws.starBlackbodySimple(distance=1.5e11,
        						latitude=0,
                                longitude=solar_zenith_angle)
    elif star_type=='Spectrum':
        gf2 = pyarts.xml.load(f"{exp_setup.arts_data_path}arts-xml-data/star/Sun/solar_spectrum.xml")
        ws.star_spectrum_raw = gf2
        ws.starFromGrid(temperature=0,
                        distance=1.5e11,
                        latitude=0,
                        longitude=solar_zenith_angle)
    elif star_type=='White': # white in frequency
        ws.starBlackbodySimple(distance=1.5e11,
        						latitude=0,
                                longitude=solar_zenith_angle)
        ws.stars.value[0].spectrum[:,0] = np.ones(np.shape(ws.stars.value[0].spectrum[:,0])) * np.max(ws.stars.value[0].spectrum)

    star_spectrum = np.array(ws.stars.value[0].spectrum)[:,0].copy()

    return ws, star_spectrum

def main():
    exp = read_exp_setup(exp_name='solar_angle', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    print(exp)
    run_arts_batch(exp)

if __name__ == '__main__':
    main()