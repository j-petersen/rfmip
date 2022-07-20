import pyarts
import requests
import numpy as np
import typhon as ty
import pandas as pd


def readin_nc(filename, fields=None):
    fh = ty.files.NetCDF4()
    if fields is None:
        return fh.read(filename)
    return fh.read(filename, fields)


def write_xml(data, filename) -> None:
    path = "/Users/jpetersen/rare/rfmip/input/"
    pyarts.xml.save(data, filename=path + filename, format="ascii")


def main() -> None:
    filename = "/Users/jpetersen/rare/rfmip/input/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    data = readin_nc(filename)
    data = data.isel(expt=0)
    sensor_pos = np.stack((data.lat.values, data.lon.values), axis=-1)
    write_xml(sensor_pos, "sensor_pos.xml")
    write_xml(data.solar_zenith_angle.values, "solar_zenith_angle.xml")
    write_xml(data.total_solar_irradiance.values, "total_solar_irradiance.xml")
    write_xml(data.surface_albedo.values, "surface_albedo.xml")
    write_xml(data.profile_weight.values, "profil_weight.xml")
    write_xml(data.surface_temperature.values, "surface_temperature.xml")
    write_xml(data.pres_layer.values[:, ::-1], 'pressure_layer.xml')
    field_names = ["T", "z"]
    spec_dict = species_name_mapping()
    spec_values = [spec_dict[key] for key in spec_dict if spec_dict[key] is not None]
    spec_keys = [key for key in spec_dict if spec_dict[key] is not None]
    field_names += spec_values

    arr_gf4 = pyarts.arts.ArrayOfGriddedField4()
    surface_elevation_arr = np.zeros(data.dims['site'])
    height_arr = np.zeros((data.dims['site'], data.dims['layer']))
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
                    data.isel(site=site)[spec].values, data.dims["layer"]
                ).astype(np.float64)
            else:
                arr[i + id_offset, :, 0, 0] = (
                    data.isel(site=site)[spec]
                    .values.reshape(data.dims["layer"])
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

    write_xml(height_arr, "additional_data/heights.xml")
    write_xml(spec_values, "species.xml")
    write_xml(surface_elevation_arr, "surface_altitudes.xml")
    write_xml(arr_gf4, "atm_fields.xml")


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


if __name__ == "__main__":
    main()
