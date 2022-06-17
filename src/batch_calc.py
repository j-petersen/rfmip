import pyarts
import numpy as np
import typhon as ty
import matplotlib.pyplot as plt
from experiment_setup import read_exp_setup

def run_arts_batch(exp_setup, verbosity=2):
    """Run Arts Calculation for RFMIP.
    """
    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")

def main():
    exp = read_exp_setup(exp_name='test')
    print(exp)
    run_arts_batch(exp)

if __name__ == '__main__':
    main()