import sys
from mazsim.cli import CLI
from mazsim.cli import simulate
from mazsim.cli import estimate
from mazsim.cli import calibrate

from mazsim import __version__, __doc__

def main():
    run_model = CLI(version=__version__, description=__doc__)
    # run_model.add_subcommand(
    #     name="simulate",
    #     args_func=simulate.add_run_args,
    #     exec_func=simulate.run,
    #     description=simulate.run.__doc__,
    # )

    run_model.add_subcommand(
        name="estimate",
        args_func=estimate.add_run_args,
        exec_func=estimate.run,
        description=estimate.run.__doc__,
    )

    # run_model.add_subcommand(
    #     name="calibrate",
    #     args_func=calibrate.add_run_args,
    #     exec_func=calibrate.run,
    #     description=calibrate.run.__doc__,
    # )

    sys.exit(run_model.execute())