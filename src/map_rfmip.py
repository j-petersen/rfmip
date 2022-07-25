# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import typhon


def get_atm_data():
    path = "/Users/froemer/Documents/rte-rrtmgp/examples/rfmip-clear-sky"
    atm = xr.open_dataset(
        f"{path}/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    )
    return atm


def map(lon, lat, color="red"):
    plt.get_cmap("density")
    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(bottom=0, top=0.9, left=0, right=0.9, wspace=0.05, hspace=0)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.set_global()
    ax.coastlines()
    cb = ax.scatter(
        x=lon, y=lat, c=color, cmap="density", s=100, transform=ccrs.PlateCarree()
    )
    cb = plt.colorbar(cb, cmap="density", orientation="vertical", label="label")


def main():
    atm = get_atm_data()
    map(atm.lon.data, atm.lat.data, atm.surface_temperature[0])


if __name__ == "__main__":
    main()

# %%
