import glob
import numpy as np
from netCDF4 import Dataset


def read_netcdf(filelist, dimensions=['all'], variables=['all']):
    datalist = []
    for file in filelist:
        data = {}
        with Dataset(file, mode='r') as f:
            if 'None' in dimensions:
                pass
            elif 'all' in dimensions:
                for dim in f.dimensions.keys():
                    data[dim] = f.dimensions[dim].__len__()
            else:
                for dim in dimensions:
                    data[dim] = f.dimensions[dim].__len__()
            
            if 'None' in variables:
                pass
            elif 'all' in variables:
                for var in f.variables.keys():
                    data[var] = f.variables[var][:]
            else:
                for var in variables:
                    data[var] = f.variables[var][:]
        datalist.append(data)

    return datalist

def get_netcdf_in_folder(path):
    """Returns a list with all .nc files in given directory. """
    file_list = []
    for name in glob.glob(f"{path}*.nc"):
        file_list.append(name)
    filelist = file_list.sort()  # sort alphabetically
    return file_list


def main():
    path = "/Users/jpetersen/rare/run_arts/rfmip_analysis/data/"
    filelist = get_netcdf_in_folder(path)
    datalist = read_netcdf(filelist)
    print(datalist)


if __name__ == '__main__':
    main()