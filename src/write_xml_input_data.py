import os
import pyarts
import requests
import numpy as np
import typhon as ty
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from urllib.request import urlretrieve

from experiment_setup import read_exp_setup


def create_input_data(exp_setup) -> None:
    if not os.path.exists(f'{exp_setup.rfmip_path}{exp_setup.input_folder}'):
        os.mkdir(f'{exp_setup.rfmip_path}{exp_setup.input_folder}')

    rfmip_data_name = "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    fetch_official_rfmip_data(exp_setup, rfmip_data_name)
    filename = f"{exp_setup.rfmip_path}input/{rfmip_data_name}"

    data = readin_nc(filename)
    data = data.isel(expt=0)

    sensor_pos = np.stack((data.lat.values, data.lon.values), axis=-1)
    write_xml(sensor_pos, "sensor_pos.xml", exp_setup)
    write_xml(data.solar_zenith_angle.values, "solar_zenith_angle.xml", exp_setup)
    write_xml(data.total_solar_irradiance.values, "total_solar_irradiance.xml", exp_setup)
    scaled_solar_spectrum(exp_setup=exp_setup)
    write_xml(data.surface_albedo.values, "surface_albedo.xml", exp_setup)
    write_xml(data.profile_weight.values, "profil_weight.xml", exp_setup)
    write_xml(data.surface_temperature.values, "surface_temperature.xml", exp_setup)
    write_xml(data.pres_layer.values[:, ::-1], "pressure_layer.xml", exp_setup)
    write_xml(data.pres_level.values[:, ::-1], "pressure_level.xml", exp_setup)
    write_xml(data.temp_level.values[:, ::-1], "temperature_level.xml", exp_setup)
    write_xml(data.temp_layer.values[:, ::-1], "temperature_layer.xml", exp_setup)

    pos = np.zeros((data.dims["site"], 2)) 
    pos[:, 0], pos[:, 1] = data.lat.values, data.lon.values
    write_xml(pos, "site_pos.xml", exp_setup)
    if exp_setup.name == 'rfmip_lvl':
        write_AtmFieldCompact_highres(exp_setup=exp_setup, data=data)
    else:
        write_AtmFieldCompact(exp_setup=exp_setup, data=data)
    
    calculate_3D_sza(exp_setup=exp_setup)


def readin_nc(filename, fields=None):
    fh = ty.files.NetCDF4()
    if fields is None:
        return fh.read(filename)
    return fh.read(filename, fields)


def write_xml(data, filename, exp_setup) -> None:
    path = f"{exp_setup.rfmip_path}{exp_setup.input_folder}"
    pyarts.xml.save(data, filename=path + filename, format="ascii")


def fetch_official_rfmip_data(exp_setup, rfmip_data_name):
    "fetches the original rfmip data if its not already present."
    filename = f"{exp_setup.rfmip_path}input/{rfmip_data_name}"
    if os.path.exists(filename):
        print("original data is already present")
        return

    url = (
        "http://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/"
        "RFMIP/UColorado/UColorado-RFMIP-1-2/atmos/fx/multiple/none/v20190401/"
        f"{rfmip_data_name}"
    )

    urlretrieve(url, filename)


def select_species(exp_setup) -> dict:
    if exp_setup.species == ["all"]:
        return species_name_mapping()

    spec_mapping = species_name_mapping()
    spec_dict = {}
    for spec in exp_setup.species:
        spec_dict[spec] = spec_mapping[spec]
    return spec_dict


def species_name_mapping() -> dict:
    name_map = {
        "oxygen_GM": "abs_species-O2",
        "nitrogen_GM": "abs_species-N2",
        "water_vapor": "abs_species-H2O",
        "ozone": "abs_species-O3",
        "carbon_monoxide_GM": "abs_species-CO",
        "c2f6_GM": "abs_species-C2F6",
        "c3f8_GM": "abs_species-C3F8",
        "c4f10_GM": "abs_species-C4F10",
        "c5f12_GM": "abs_species-C5F12",
        "c6f14_GM": "abs_species-C6F14",
        "c7f16_GM": None,
        "c8f18_GM": "abs_species-C8F18",
        "c_c4f8_GM": "abs_species-cC4F8",
        "carbon_dioxide_GM": "abs_species-CO2",
        "carbon_tetrachloride_GM": "abs_species-CCl4",
        "cf4_GM": "abs_species-CF4",
        "cfc113_GM": "abs_species-CFC113",
        "cfc114_GM": "abs_species-CFC114",
        "cfc115_GM": "abs_species-CFC115",
        "cfc11_GM": "abs_species-CFC11",
        "cfc11eq_GM": None,
        "cfc12_GM": "abs_species-CFC12",
        "cfc12eq_GM": None,
        "ch2cl2_GM": "abs_species-CH2Cl2",
        "ch3ccl3_GM": "abs_species-CH3CCl3",
        "chcl3_GM": "abs_species-CHCl3",
        "halon1211_GM": "abs_species-Halon1211",
        "halon1301_GM": "abs_species-Halon1301",
        "halon2402_GM": "abs_species-Halon2402",
        "hcfc141b_GM": "abs_species-HCFC141b",
        "hcfc142b_GM": "abs_species-HCFC142b",
        "hcfc22_GM": "abs_species-HCFC22",
        "hfc125_GM": "abs_species-HFC125",
        "hfc134a_GM": "abs_species-HFC134a",
        "hfc134aeq_GM": None,
        "hfc143a_GM": "abs_species-HFC143a",
        "hfc152a_GM": "abs_species-HFC152a",
        "hfc227ea_GM": "abs_species-HFC227ea",
        "hfc236fa_GM": None,
        "hfc23_GM": "abs_species-HFC23",
        "hfc245fa_GM": None,
        "hfc32_GM": "abs_species-HFC32",
        "hfc365mfc_GM": None,
        "hfc4310mee_GM": None,
        "methane_GM": "abs_species-CH4",
        "methyl_bromide_GM": "abs_species-CH3Br",
        "methyl_chloride_GM": "abs_species-CH3Cl",
        "nf3_GM": "abs_species-NF3",
        "nitrous_oxide_GM": "abs_species-N2O",
        "sf6_GM": "abs_species-SF6",
        "so2f2_GM": "abs_species-SO2F2",
    }
    return name_map


def interpolate_data_on_lvl(spec_field, site, spec, exp_setup):
    pres_layer = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}pressure_layer.xml")[site, ::-1]
    pres_lvl = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}pressure_level.xml")[site, ::-1]

    pres_high_res = np.zeros((len(pres_lvl) + len(pres_layer)))
    pres_high_res[::2] = pres_lvl
    pres_high_res[1::2] = pres_layer

    new_p_grid = pres_high_res if exp_setup.name == 'rfmip_lvl' else pres_lvl

    f = interpolate.interp1d(np.log(pres_layer), spec_field[::-1], kind='linear', fill_value="extrapolate")
    interp_spec_field = f(np.log(new_p_grid))[::-1]
    interp_spec_field[interp_spec_field < 0] = 0

    # ty.plots.styles.use(['typhon', 'typhon-dark'])
    # fig, ax = plt.subplots(figsize=(12, 9))
    
    # ax.plot(spec_field, pres_layer[::-1], label='raw')
    # ax.plot(interp_spec_field, pres_lvl[::-1], label='interp')
    
    # ax.set_xlabel('spec')
    # ax.set_ylabel('pressure')
    
    # ax.legend(frameon=False)
    # fig.savefig(f'/Users/jpetersen/rare/rfmip/plots/analysis/interp/interp_site_{site}_{spec}.png', dpi=200)
    # plt.close("all")

    return interp_spec_field


def write_AtmFieldCompact(exp_setup, data):
    field_names = ["T", "z"]
    spec_dict = select_species(exp_setup)
    spec_values = [spec_dict[key] for key in spec_dict if spec_dict[key] is not None]
    spec_keys = [key for key in spec_dict if spec_dict[key] is not None]
    field_names += spec_values  

    arr_gf4 = pyarts.arts.ArrayOfGriddedField4()
    surface_elevation_arr = np.zeros(data.dims["site"])
    level_height = np.zeros((data.dims["site"], data.dims["level"]))
    
    for site in range(data.dims["site"]):
        arr = np.zeros(
            (len(field_names), data.dims["level"], 1, 1)
        )
        arr[0, :, 0, 0] = data.isel(site=site).temp_level.values[::-1]

        z_levels = ty.physics.pressure2height(
            data.isel(site=site).pres_level.values[::-1],
            data.isel(site=site).temp_level.values[::-1],
        )

        z_elevation = get_elevation(
            [[data.isel(site=site).lat.values, data.isel(site=site).lon.values]]
        )
        surface_elevation_arr[site] = z_elevation

        arr[1, :, 0, 0] = z_levels + z_elevation
        level_height[site] = z_levels + z_elevation

        id_offset = 2
        for i, spec in enumerate(spec_keys):
            if np.shape(data.isel(site=site)[spec].values) == ():
                spec_field = np.tile(
                    data.isel(site=site)[spec].values
                    * np.float64(data.isel(site=site)[spec].attrs["units"]),
                    data.dims["level"],
                ).astype(np.float64)
            else:
                spec_field_raw = (
                    (
                        data.isel(site=site)[spec].values
                        * np.float64(data.isel(site=site)[spec].attrs["units"])
                    )
                    .reshape(data.dims["layer"])
                    .astype(np.float64)
                )[::-1]
                spec_field = interpolate_data_on_lvl(spec_field_raw, site, spec, exp_setup=exp_setup)

            arr[i + id_offset, :, 0, 0] = spec_field
        gf4 = pyarts.arts.GriddedField4(
            grids=[
                field_names,
                data.isel(site=site)
                .pres_level.values.reshape(data.dims["level"])
                .astype(np.float64)[::-1],
                [],
                # data.isel(site=site).lat.values.reshape(1).astype(np.float64),
                []
                # data.isel(site=site).lon.values.reshape(1).astype(np.float64)
            ],
            data=arr,
            gridnames=["field_names", "p_grid", "lat_grid", "lon_grid"],
            name=f"site_{site}",
        )
        arr_gf4.append(gf4)

    write_xml(level_height, "height_levels.xml", exp_setup)
    write_xml(spec_values, "species.xml", exp_setup)
    write_xml(surface_elevation_arr, "surface_altitudes.xml", exp_setup)
    write_xml(arr_gf4, "atm_fields.xml", exp_setup)


def write_AtmFieldCompact_highres(exp_setup, data):
    field_names = ["T", "z"]
    spec_dict = select_species(exp_setup)
    spec_values = [spec_dict[key] for key in spec_dict if spec_dict[key] is not None]
    spec_keys = [key for key in spec_dict if spec_dict[key] is not None]
    field_names += spec_values  

    arr_gf4 = pyarts.arts.ArrayOfGriddedField4()
    surface_elevation_arr = np.zeros(data.dims["site"])
    n_lvl = data.dims["level"]+data.dims['layer']
    level_height = np.zeros((data.dims["site"], n_lvl))
    pressure = np.zeros((data.dims["site"], n_lvl))

    for site in range(data.dims["site"]):
        arr = np.zeros(
            (len(field_names), n_lvl, 1, 1)
        )
        arr[0, ::2, 0, 0] = data.isel(site=site).temp_level.values[::-1]
        arr[0, 1::2, 0, 0] = data.isel(site=site).temp_layer.values[::-1]

        pressure[site, ::2] = data.isel(site=site).pres_level.values[::-1]
        pressure[site, 1::2] = data.isel(site=site).pres_layer.values[::-1]

        z_levels = ty.physics.pressure2height(
            pressure[site],
            arr[0, :, 0, 0],
        )

        z_elevation = get_elevation(
            [[data.isel(site=site).lat.values, data.isel(site=site).lon.values]]
        )
        surface_elevation_arr[site] = z_elevation

        arr[1, :, 0, 0] = z_levels + z_elevation
        level_height[site] = z_levels + z_elevation

        id_offset = 2
        for i, spec in enumerate(spec_keys):
            if np.shape(data.isel(site=site)[spec].values) == ():
                spec_field = np.tile(
                    data.isel(site=site)[spec].values
                    * np.float64(data.isel(site=site)[spec].attrs["units"]),
                    n_lvl,
                ).astype(np.float64)
            else:
                spec_field_raw = (
                    (
                        data.isel(site=site)[spec].values
                        * np.float64(data.isel(site=site)[spec].attrs["units"])
                    )
                    .reshape(data.dims["layer"])
                    .astype(np.float64)
                )[::-1]
                spec_field = interpolate_data_on_lvl(spec_field_raw, site, spec, exp_setup=exp_setup)

            arr[i + id_offset, :, 0, 0] = spec_field
        gf4 = pyarts.arts.GriddedField4(
            grids=[
                field_names,
                pressure[site],
                [],
                # data.isel(site=site).lat.values.reshape(1).astype(np.float64),
                []
                # data.isel(site=site).lon.values.reshape(1).astype(np.float64)
            ],
            data=arr,
            gridnames=["field_names", "p_grid", "lat_grid", "lon_grid"],
            name=f"site_{site}",
        )
        arr_gf4.append(gf4)
    write_xml(level_height, "height_levels.xml", exp_setup)
    write_xml(pressure, "pressure_level.xml", exp_setup)
    write_xml(spec_values, "species.xml", exp_setup)
    write_xml(surface_elevation_arr, "surface_altitudes.xml", exp_setup)
    write_xml(arr_gf4, "atm_fields.xml", exp_setup)


# script for returning elevation from lat, long, based on open elevation data
# which in turn is based on SRTM
# https://stackoverflow.com/questions/19513212/can-i-get-the-altitude-with-geopy-in-python-with-longitude-latitude
def get_elevation(geo_data=None):
    if geo_data is None:
        geo_data = pyarts.xml.load(
            "/Users/jpetersen/rare/rfmip/input/sensor_pos.xml"
        )

    geo_str = ""
    for pos in geo_data:
        geo_str += f"{pos[0]},{pos[1]}|"
    query = "https://api.open-elevation.com/api/v1/lookup" f"?locations={geo_str[:-1]}"
    r = requests.get(query).json()  # json object, various ways you can extract value
    # one approach is to use pandas json functionality:
    elevation = np.array(pd.json_normalize(r, "results")["elevation"].values)
    return elevation


def calculate_3D_sza(exp_setup) -> None:
    solar_zenith_angles = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}solar_zenith_angle.xml")
    heights = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}height_levels.xml")
    star_distance = 1.495978707e11
    earth_radius = 6.3781e6

    solar_pos = np.zeros((len(solar_zenith_angles)))
    for i, sza in enumerate(solar_zenith_angles):
        solar_pos[i] = sza - np.rad2deg(np.arcsin((earth_radius+heights[i, -1])/star_distance * np.sin(np.deg2rad(180-sza))))

    write_xml(solar_pos, "solar_pos.xml", exp_setup)

def scaled_solar_spectrum(exp_setup) -> None:
    total_solar_irradiances = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}total_solar_irradiance.xml")
    gf2 = pyarts.xml.load(f"{exp_setup.arts_data_path}arts-xml-data/star/Sun/solar_spectrum_May_2004.xml")

    arr_gf2 = pyarts.arts.ArrayOfGriddedField2()
    for i, tsi in enumerate(total_solar_irradiances):
        arr_gf2.append(scale2tsi(exp_setup, gf2, tsi=tsi, idx=-1)) # no offset - chose i if you want the offset

    write_xml(arr_gf2, "star_spectra.xml", exp_setup)


def scale2tsi(exp_setup, spectrum, tsi=1366, at_toa=False, idx=-1):
    if not at_toa:
        offset = 0
        if idx != -1:
            offset = get_site_star_distance_offset(exp_setup=exp_setup)[idx]
        spectrum.data = irradstar2irradToa(spectrum.data, distance_offset=offset)
    solar_f_grid = spectrum.grids[0]
    solar_spec = spectrum.data[:, 0]
    sim_f_grid = f_grid_from_spectral_grid(exp_setup=exp_setup)

    f = interpolate.interp1d(solar_f_grid, solar_spec, kind='linear')
    interp_spec = f(sim_f_grid)

    spec_tsi = np.trapz(interp_spec, sim_f_grid)
    
    spectrum.data = spectrum.data * tsi/spec_tsi
    # if not at_toa:
        # spectrum.data = irradTOA2irradstar(spectrum.data)
    return spectrum


def irradstar2irradToa(irradiance, star_radius=6.963242e8, star_distance=1.495978707e11, distance_offset=0):
    """ Converts radiance at star surface to irradance at TOA"""
    earth_radius =  6_378_000
    star_distance -= (100_000 + earth_radius + distance_offset)
    # accounts for the distance
    factor = star_radius**2/(star_radius**2 + star_distance**2)

    return irradiance * factor


def irradTOA2irradstar(irradiance, star_radius=6.963242e8, star_distance=1.495978707e11):
    """ Converts radiance at TOA to irradance at star surface"""
    earth_radius =  6_378_000
    star_distance -= (100_000 + earth_radius)

    # accounts for the distance
    factor = star_radius**2/(star_radius**2 + star_distance**2)

    return irradiance / factor

def get_site_star_distance_offset(exp_setup, earth_radius=6.378e6):
    heights = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}heights.xml")
    sza = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}solar_zenith_angle.xml")
    toa_height = heights[:, -1]

    offset = 1e5-toa_height # from toa definition
    offset += np.sin(np.deg2rad(sza)) * (earth_radius+toa_height)
    return offset


def f_grid_from_spectral_grid(exp_setup):
    if exp_setup.which_spectral_grid == 'frequency':
        f_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)
    elif exp_setup.which_spectral_grid == 'wavelength':
        lam_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)*1e-9
        f_grid = ty.physics.wavelength2frequency(lam_grid)[::-1]
    elif exp_setup.which_spectral_grid == 'kayser':
        kayser_grid = np.linspace(exp_setup.spectral_grid['min'], exp_setup.spectral_grid['max'], exp_setup.spectral_grid['n'], endpoint=True)*1e2
        f_grid = ty.physics.wavenumber2frequency(kayser_grid)

    return f_grid

def main():
    exp_setup = read_exp_setup(
        exp_name="rfmip_lvl", path="/Users/jpetersen/rare/rfmip/experiment_setups/",
    )
    create_input_data(exp_setup=exp_setup)
    # calculate_3D_sza(exp_setup)


if __name__ == "__main__":
    main()
