import re
import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt


def main():
    ty.plots.styles.use(['typhon', 'typhon-dark'])
    site = [11, 54, 83]#, 75]
    # plot_temperature_profile(site)
    plot_vmrs(site)
    plt.show()

def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)

def plot_temperature_profile(sites):
    species = ["N2"]
    z_fields = []
    t_fields = []
    for site in sites:
        z_field, t_field, _ =  read_atomsphere(species, site=site)
        z_fields.append(z_field)    
        t_fields.append(t_field)

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, site in enumerate(sites):    
        ax.plot(t_fields[i], z_fields[i] / 1000, label=site)
    
    ax.set_xlabel('temperature / K')
    ax.set_ylabel('altitude / km')
    
    ax.legend(frameon=False)
    fig.savefig(f"/Users/jpetersen/rare/rfmip/plots/analysis/atm/temperature_site{'_'.join(str(site) for site in sites)}_dark.png", dpi=200)


def plot_vmrs(sites):
    species = ["H2O"]
    z_fields = []
    vmrs = []
    for site in sites:
        z_field, _,  vmr = read_atomsphere(species, site=site)
        z_fields.append(z_field)
        vmrs.append(vmr)

    fig, ax = plt.subplots(figsize=(12, 9))
    print(vmrs)
    
    for i, site in enumerate(sites):
        label = re.split(', |-', species[0])
        line = ax.semilogx(vmrs[i][:-3] * 100, z_fields[i][:-3] / 1000, label=f'site: {site}') # ({tag2tex(label[0])})
        # ax.semilogx(vmrs[i][1] * 100, z_fields[i] / 1000, linestyle='--', color=line[0].get_color())
    
    ax.set_xlabel(f'{tag2tex(label[0])} vmr / %', fontsize=24)
    ax.set_ylabel('altitude / km', fontsize=24)
    # ax.tick_params(size=14)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    
    ax.legend(frameon=False)
    fig.savefig(f'/Users/jpetersen/rare/rfmip/plots/analysis/atm/vmrs_site{"_".join(str(site) for site in sites)}_dark.png', dpi=200)


def read_atomsphere(species, site):
    ws = pyarts.workspace.Workspace()

    ws.AtmosphereSet1D()
    ws.abs_speciesSet(species=species)
    
    ws.batch_atm_fields_compact = pyarts.xml.load('/Users/jpetersen/rare/rfmip/input/rfmip/atm_fields.xml')
    ws.Extract(ws.atm_fields_compact, ws.batch_atm_fields_compact, site)

    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    t_field = np.squeeze(ws.t_field.value)
    z_field = np.squeeze(ws.z_field.value)
    vmrs = np.squeeze(ws.vmr_field.value)

    return z_field, t_field, vmrs
    


if __name__ == '__main__':
    main()