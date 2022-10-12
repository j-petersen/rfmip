""" This is a sript so automate and document the calling of the sbatch script to start ARTS simulations on levante. """
import os

SBATCHSCRIPT = "/work/um0878/sw/bin/sbatch_simple"
PARTITION = "compute"
JOBNAME = "arts_sim"
NCORES = 64
TIMELIMIT = "08:00:00"


def run_experiments(exp_setup):
    if isinstance(exp_setup, str):
        exp_setup = [exp_setup]

    for exp in exp_setup:
        if not os.path.exists(f'{os.getcwd()}/experiment_setups/{exp}.json'):
            with FileNotFoundError as e:
                e.args = (f"The experiment '{exp}' does not exist in the experiment_setups folder!",)
                raise e
        
        command = f"python3 src/main.py {exp}"
        shell_command = (
            f"{SBATCHSCRIPT} {PARTITION} arts_{exp} {NCORES} {TIMELIMIT} {command}"
        )
        print(f"{shell_command=}")
        os.system(command=shell_command)


def calc_luts(exp_setup, n_chunks=8):
    if isinstance(exp_setup, str):
        exp_setup = [exp_setup]

    for exp in exp_setup:
        if not os.path.exists(f'{os.getcwd()}/experiment_setups/{exp}.json'):
            with FileNotFoundError as e:
                e.args = (f"The experiment '{exp}' does not exist in the experiment_setups folder!",)
                raise e
        for i in range(nchunks):
            command = f"python3 src/batch_lookuptable.py -e {exp} -c {n_chunks} {i}"
            shell_command = (
                f"{SBATCHSCRIPT} {PARTITION} lut{i}_{exp} {NCORES} {TIMELIMIT} {command}"
            )
            print(f"{shell_command=}")
            os.system(command=shell_command)

def combine_luts(exp_setup, n_chunks=8):
    if isinstance(exp_setup, str):
        exp_setup = [exp_setup]

    for exp in exp_setup:
        if not os.path.exists(f'{os.getcwd()}/experiment_setups/{exp}.json'):
            with FileNotFoundError as e:
                e.args = (f"The experiment '{exp}' does not exist in the experiment_setups folder!",)
                raise e
        command = f"python3 src/batch_lookuptable.py -e {exp} -c {n_chunks} {0} --combine"
        shell_command = (
            f"{SBATCHSCRIPT} {PARTITION} lutcom_{exp} {NCORES} {TIMELIMIT} {command}"
        )
        print(f"{shell_command=}")
        os.system(command=shell_command)

def main():
    exp_setups = ['rfmip', 'rfmip_no_star', 'rfmip_no_emission']
    calc_luts(exp_setup=exp_setups[0], n_chunks=32)
    combine_luts(exp_setup=exp_setups[0], n_chunks=32)
    run_experiments(exp_setup=exp_setups)


if __name__ == "__main__":
    main()
