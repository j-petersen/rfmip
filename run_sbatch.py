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
            f"{SBATCHSCRIPT} {PARTITION} {JOBNAME} {NCORES} {TIMELIMIT} {command}"
        )
        print(f"{shell_command=}")
        os.system(command=shell_command)


def calc_luts(exp_setup):
    if isinstance(exp_setup, str):
        exp_setup = [exp_setup]

    for exp in exp_setup:
        if not os.path.exists(f'{os.getcwd()}/experiment_setups/{exp}.json'):
            with FileNotFoundError as e:
                e.args = (f"The experiment '{exp}' does not exist in the experiment_setups folder!",)
                raise e
        
        command = f"python3 src/batch_lookuptable.py {exp}"
        shell_command = (
            f"{SBATCHSCRIPT} {PARTITION} {JOBNAME} {NCORES} {TIMELIMIT} {command}"
        )
        print(f"{shell_command=}")
        os.system(command=shell_command)


def main():
    exp_setups = ['rfmip', 'rfmip_no_star', 'rfmip_no_emission']
    calc_luts(exp_setup=exp_setups[0])
    run_experiments(exp_setup=exp_setups)


if __name__ == "__main__":
    main()
