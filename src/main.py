import pyarts

import write_xml_input_data as input_data
import experiment_setup as setup
import batch_calc as calc
import data_processing as post_pro
import data_visualisation as vis

def main():
    # Write experiment setup
    setup.new_test_setup()
    exp_setup = setup.read_exp_setup(exp_name='test', path='/Users/jpetersen/rare/rfmip/experiment_setups/')
    
    # Create input data
    print('Create input data')
    input_data.create_input_data(exp_setup)

    # Lookuptable (not yet implemented)
    # ToDo seperate lookuptable calc from main calculation

    # Calculation
    print('Calculation')
    calc.run_arts_batch(exp_setup)

    # Postprocessing
    print('Postprocessing')
    data = post_pro.read_spectral_irradiance(exp_setup)
    data = post_pro.combine_sites(data, exp_setup)
    post_pro.save_data(data, exp_setup)

    # Visualisation
    print('Visualisation')
    vis.plot_flux_profiles(exp_setup=exp_setup)


if __name__ == '__main__':
    main()