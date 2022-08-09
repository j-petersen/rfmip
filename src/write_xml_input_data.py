import os
import pyarts
import requests
import numpy as np
import typhon as ty
import pandas as pd
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

    field_names = ["T", "z"]
    spec_dict = select_species(exp_setup)
    spec_values = [spec_dict[key] for key in spec_dict if spec_dict[key] is not None]
    spec_keys = [key for key in spec_dict if spec_dict[key] is not None]
    field_names += spec_values

    arr_gf4 = pyarts.arts.ArrayOfGriddedField4()
    surface_elevation_arr = np.zeros(data.dims["site"])
    height_arr = np.zeros((data.dims["site"], data.dims["layer"]))
    for site in range(data.dims["site"]):
        arr = np.zeros(
            (len(field_names), len(data.isel(site=0).pres_layer.values), 1, 1)
        )
        arr[0, :, 0, 0] = data.isel(site=site).temp_layer.values[::-1]

        z_above_ground = ty.physics.pressure2height(
            data.isel(site=site).pres_layer.values[::-1],
            data.isel(site=site).temp_layer.values[::-1],
        )

        z_elevation = get_elevation(
            [[data.isel(site=site).lat.values, data.isel(site=site).lon.values]]
        )
        surface_elevation_arr[site] = z_elevation

        arr[1, :, 0, 0] = z_above_ground + z_elevation
        height_arr[site] = z_above_ground + z_elevation

        id_offset = 2
        for i, spec in enumerate(spec_keys):
            if np.shape(data.isel(site=site)[spec].values) == ():
                arr[i + id_offset, :, 0, 0] = np.tile(
                    data.isel(site=site)[spec].values
                    * np.float64(data.isel(site=site)[spec].attrs["units"]),
                    data.dims["layer"],
                ).astype(np.float64)
            else:
                arr[i + id_offset, :, 0, 0] = (
                    (
                        data.isel(site=site)[spec].values
                        * np.float64(data.isel(site=site)[spec].attrs["units"])
                    )
                    .reshape(data.dims["layer"])
                    .astype(np.float64)
                )[::-1]

        gf4 = pyarts.arts.GriddedField4(
            grids=[
                field_names,
                data.isel(site=site)
                .pres_layer.values.reshape(data.dims["layer"])
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

    write_xml(height_arr, "heights.xml", exp_setup)
    write_xml(spec_values, "species.xml", exp_setup)
    write_xml(surface_elevation_arr, "surface_altitudes.xml", exp_setup)
    write_xml(arr_gf4, "atm_fields.xml", exp_setup)


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
        "c7f16_GM": "abs_species-C7F16",
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


def scaled_solar_spectrum(exp_setup) -> None:
    total_solar_irradiances = pyarts.xml.load(f"{exp_setup.rfmip_path}{exp_setup.input_folder}total_solar_irradiance.xml")
    gf2 = pyarts.xml.load(f"{exp_setup.arts_data_path}arts-xml-data/star/Sun/solar_spectrum.xml")
    arr_gf2 = pyarts.arts.ArrayOfGriddedField2()

    sigma = 5.670375419e-8
    star_radius = 6.96342e8
    star_distance = 1.495978707e11
    star_effective_temperature = 5772
    alpha = np.arctan2(star_radius, star_distance)
    tsi_spectrum = sigma * star_effective_temperature**4 * np.sin(alpha)**2
    
    for tsi in total_solar_irradiances:
        gf2_scaled = gf2
        sigma * 5772**4 * np.sin(alpha)**2
        gf2_scaled.data = gf2_scaled.data * tsi/tsi_spectrum
        arr_gf2.append(gf2_scaled)

    write_xml(arr_gf2, "star_spectra.xml", exp_setup)


def main():
    exp_setup = read_exp_setup(
        exp_name="olr", path="/Users/jpetersen/rare/rfmip/experiment_setups/",
    )
    create_input_data(exp_setup=exp_setup)


if __name__ == "__main__":
    main()
