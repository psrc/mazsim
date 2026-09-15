import orca
import sys
from pathlib import Path
import yaml
import zipfile


@orca.step('save_output_summaries')
def save_output_summaries():
    output_dir = Path.joinpath(orca.get_injectable('project_dir'),orca.get_injectable('output_dir'),'output_summaries')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(Path.joinpath(orca.get_injectable('project_dir'),'configs','output_summary.yaml')))
    year = orca.get_injectable('year')
    variables = cfg['variables']
    for geography in cfg['geography']:
        df = orca.get_table(geography).to_frame(variables)
        df.to_csv(Path.joinpath(output_dir, f'{geography}_{year}.csv'))


class _Tee:
    """Forwards writes to the original stream and to the run log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._stream.isatty()

@orca.step('start_run_log')
def start_run_log(project_dir):
    """Capture everything printed to stdout/stderr in output/run_N.log."""
    output_dir = Path(project_dir) / orca.get_injectable('output_dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    run_number = get_last_run_number(project_dir) + 1
    orca.add_injectable("run_number", run_number)
    log_path = output_dir / f"run_{run_number}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_path


def get_last_run_number(project_dir):
    """Highest N found in output/results_N.h5, or 0 when no runs exist yet."""
    output_dir = Path(project_dir) / orca.get_injectable('output_dir') / "archive"
    run_numbers = []
    for path in output_dir.glob("results_*.zip"):
        suffix = path.stem.split("_")[-1]
        if suffix.isdigit():
            run_numbers.append(int(suffix))
    return max(run_numbers, default=0)


@orca.step('save_full_tables')
def save_full_tables():
    year = orca.get_injectable('year')
    output_tables = orca.get_injectable('output_tables')
    project_dir = orca.get_injectable('project_dir')
    run_number = orca.get_injectable('run_number')
    export_h5 = Path.joinpath(project_dir, "output", f"results_{run_number}.h5")
    output_path = Path.joinpath(project_dir, "output")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving results tables to {export_h5}")
    for table_name in output_tables:
        df = orca.get_table(table_name).local
        df.to_hdf(export_h5, key=f"{year}/{table_name}", mode="a")


@orca.step('archive_results')
def archive_results():
    project_dir = orca.get_injectable('project_dir')
    run_number = orca.get_injectable('run_number')
    output_dir = Path.joinpath(project_dir, orca.get_injectable('output_dir'))
    archive_dir = Path.joinpath(output_dir, 'archive')
    Path(archive_dir).mkdir(parents=True, exist_ok=True)
    # gather up all the summary and full result files and save them in a zip file in archive
    archive_file = archive_dir / f"results_{run_number}.zip"
    results_h5 = output_dir / f"results_{run_number}.h5"
    run_log = output_dir / f"run_{run_number}.log"
    summary_files = sorted((output_dir / 'output_summaries').glob('*.csv'))
    config_files = sorted(p for p in (project_dir / 'configs').rglob('*') if p.is_file())
    sys.stdout.flush()
    with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in (results_h5, run_log):
            if path.exists():
                zipf.write(path, path.name)
        for summary_file in summary_files:
            zipf.write(summary_file, Path('output_summaries') / summary_file.name)
        for validation_file in sorted((output_dir / 'validation_summaries').glob('*.csv')):
            zipf.write(validation_file, Path('validation_summaries') / validation_file.name)
        for config_file in config_files:
            zipf.write(config_file, Path('configs') / config_file.relative_to(project_dir / 'configs'))
    print(f"Archived results to {archive_file}")

@orca.step('delete_non_archived_run_files')
def delete_non_archived_run_files():
    project_dir = orca.get_injectable('project_dir')
    output_dir = Path.joinpath(project_dir, orca.get_injectable('output_dir'))
    summaries_dir = Path.joinpath(output_dir, 'output_summaries')
    validation_dir = Path.joinpath(output_dir, 'validation_summaries')
    run_number = orca.get_injectable('run_number')
    current_log = f'run_{run_number}.log'

    # delete h5 and logfile from previous run
    for path in output_dir.iterdir():
        if path.suffix not in ('.h5', '.log'):
            continue
        if path.name == current_log:
            continue
        path.unlink()

    # delete run summaries
    if summaries_dir.is_dir():
        for path in summaries_dir.iterdir():
            if path.is_file():
                path.unlink()

    # delete validation simmaries
        # delete run summaries
    if validation_dir.is_dir():
        for path in validation_dir.iterdir():
            if path.is_file():
                path.unlink()

