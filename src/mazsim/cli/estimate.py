"""Estimate the location choice (HLCM/JLCM/HULCM) and price (HUPM) models defined in estimate.yaml.

Every sub-model is narrowed down with forward stepwise selection on BIC before it is
registered, choosing from estimate.yaml's top-level global_expl_vars plus any
addl_expl_vars listed for its family or for the sub-model itself.
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Any, Callable
import sys

import numpy as np
import orca
import yaml
from urbansim.models import util
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import LargeMultinomialLogitStep, OLSRegressionStep
from urbansim_templates.utils import to_list

# side-effect imports: registers the load_data/build_networks/register_variables orca steps
from mazsim import data_loader, variable_loader
from mazsim.submodels import initialize_submodels


def _load_estimate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "estimate.yaml"
    return yaml.safe_load(config_path.read_text())


def _load_calibration_period(project_dir: Path) -> tuple[int, int]:
    """The base/historic years from calibrate.yaml, which fill the filter template placeholders."""
    config_path = project_dir / "configs" / "calibrate.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found; it supplies base_year and historic_year, which the "
            "sub-model filter templates are filled from."
        )
    calibrate_yaml = yaml.safe_load(config_path.read_text())
    return calibrate_yaml["base_year"], calibrate_yaml["historic_year"]


def _load_config(project_dir: Path, key: str) -> dict[str, Any]:
    return _load_estimate_yaml(project_dir)[key]


def _load_global_expl_vars(project_dir: Path) -> list[str]:
    """The candidate explanatory variables shared by every sub-model in estimate.yaml."""
    global_expl_vars = _load_estimate_yaml(project_dir).get("global_expl_vars")
    if not global_expl_vars:
        raise ValueError(
            "estimate.yaml has no global_expl_vars; it supplies the explanatory variables "
            "every sub-model's stepwise selection starts from."
        )
    return global_expl_vars


def _fit_submodel(
    config: dict[str, Any],
    model_config: dict[str, Any],
    base_year: int,
    historic_year: int,
    expl_vars: list[str],
) -> LargeMultinomialLogitStep:
    """Fit one location choice sub-model (household, job, or housing unit) with the given variables.

    The stepwise search calls this repeatedly with each candidate subset it narrows down.
    """
    filter_value = model_config["filter_value"]
    period = {"base_year": base_year, "historic_year": historic_year}

    m = LargeMultinomialLogitStep()
    m.choosers = [config["choosers"]]
    m.chooser_filters = config["chooser_filters_template"].format(value=filter_value, **period)
    m.alternatives = [config["alternatives"]]
    m.choice_column = config["choice_column"]
    m.constrained_choices = config["constrained_choices"]
    m.alt_sample_size = config["alt_sample_size"]
    m.out_chooser_filters = config["out_chooser_filters_template"].format(value=filter_value, **period)
    m.alt_capacity = config["alt_capacity"]
    m.out_alt_filters = config["out_alt_filters"]

    m.model_expression = util.str_model_expression(expl_vars, add_constant=False)
    m.fit()
    m.name = model_config["name"]  # fit() generates its own name, so overwrite it afterwards
    return m


def _fit_hupm_submodel(
    config: dict[str, Any], model_config: dict[str, Any], expl_vars: list[str]
) -> OLSRegressionStep:
    """Fit one OLS price sub-model (residential value or rent) with the given variables.

    The stepwise search calls this repeatedly with each candidate subset it narrows down.
    """
    m = OLSRegressionStep()
    m.tables = config["tables"]
    m.filters = model_config["filters"]

    model_spec = {"left_side": model_config["left_side"], "right_side": expl_vars}
    m.model_expression = util.str_model_expression(model_spec)
    m.out_filters = config["out_filters"]
    m.out_column = model_config["left_side"]
    m.fit()
    m.name = model_config["name"]  # fit() generates its own name, so overwrite it afterwards
    return m


def coeffs(m):
    """Map each fitted coefficient to the name of the variable it multiplies."""
    if isinstance(m, OLSRegressionStep):
        names = m.model.model_fit.model.exog_names
    else:
        names = m.model.results['x_names']
    return dict(zip(names, m.fitted_parameters))


def _fit_stats(m: Any) -> dict[str, Any]:
    """The model-fit statistics that stepwise ranks candidate specifications by.

    MNL steps expose choicemodels' log-likelihood table, which already carries bic, aic,
    rho_bar_squared, and df_model; OLS steps expose the equivalent statsmodels values.
    """
    if isinstance(m, OLSRegressionStep):
        fit = m.model.model_fit
        return {"bic": fit.bic, "aic": fit.aic,
                "rho_bar_squared": fit.rsquared_adj, "df_model": fit.df_model}
    return m.model.results['log_likelihood']


def stepwise(fit, candidates, max_vars=12, min_improvement=6.0, expected_sign=None, log=None):
    """Forward selection on BIC with a floating (drop) step.

    `fit` takes a list of explanatory variable names and returns a fitted model step, so
    the same search can narrow the MNL location choice and the OLS price sub-models down.
    A candidate is skipped if `expected_sign` maps its variable to the opposite sign of
    its fitted coefficient. `log`, if given, collects a dict per accepted variable.

    Returns the selected variables, the best fitted step, and its BIC.
    """
    selected, best_model = [], None
    best_bic = None
    while len(selected) < max_vars:
        trials = []
        for c in candidates:
            if c in selected:
                continue
            m = fit(selected + [c])
            stats = _fit_stats(m)
            if expected_sign and np.sign(coeffs(m)[c]) != expected_sign.get(c, np.sign(coeffs(m)[c])):
                continue  # sign violation -> ineligible
            trials.append((stats['bic'], c, m, stats))
        if not trials:
            break
        bic, c, m, stats = min(trials, key=lambda t: t[0])
        if best_bic is not None and (best_bic - bic) < min_improvement:
            break
        selected.append(c); best_model, best_bic = m, bic
        if log is not None:
            log.append({'added': c, 'bic': bic, 'aic': stats['aic'],
                        'rho_bar_squared': stats['rho_bar_squared'],
                        'n_vars': stats['df_model']})
        # floating step: try dropping each selected var once
        for d in list(selected):
            trial = [v for v in selected if v != d]
            if not trial:
                continue
            m2 = fit(trial)
            bic2 = _fit_stats(m2)['bic']
            if best_bic - bic2 > min_improvement:
                selected, best_model, best_bic = trial, m2, bic2
    return selected, best_model, best_bic


def _candidate_variables(global_expl_vars: list[str], config: dict[str, Any],
                         model_config: dict[str, Any]) -> list[str]:
    """The candidate variables one sub-model's stepwise selection starts from.

    estimate.yaml's global_expl_vars apply to every sub-model, and a family (e.g. hupm)
    or an individual sub-model entry (e.g. hupm_rent) can extend them with its own
    'addl_expl_vars' list. Duplicates are dropped, keeping their first occurrence.
    """
    addl_expl_vars = [config.get("addl_expl_vars"), model_config.get("addl_expl_vars")]
    for source in [global_expl_vars, *addl_expl_vars]:
        if source is not None and not isinstance(source, list):
            raise ValueError(
                f"estimate.yaml's explanatory variable lists must be YAML lists, got "
                f"{source!r}; check global_expl_vars and addl_expl_vars."
            )

    candidates = global_expl_vars + [v for source in addl_expl_vars if source for v in source]
    return list(dict.fromkeys(candidates))


def _check_candidates(candidates: list[str], table_names: list[str],
                      model_config: dict[str, Any]) -> None:
    """Fail with the offending names if a candidate cannot be used by this sub-model.

    Candidates come from one shared list, so a name that is not a column of the tables the
    sub-model reads, or that is empty in every one of them, would otherwise surface only as
    an obscure error inside patsy, choicemodels, or statsmodels.
    """
    problems = []
    for variable in candidates:
        sources = [name for name in table_names if variable in orca.get_table(name).columns]
        if not sources:
            problems.append(f"{variable}: not a column of {', '.join(table_names)}")
        elif all(orca.get_table(name)[variable].isna().all() for name in sources):
            problems.append(f"{variable}: all values missing in {', '.join(sources)}")
    if problems:
        raise ValueError(
            f"{model_config['name']}: estimate.yaml's candidate variables are not usable:\n"
            + "\n".join("  " + problem for problem in problems)
            + "\nregister the missing variables or remove them from global_expl_vars / "
            "addl_expl_vars."
        )


def _select_submodel(fit: Callable[[list[str]], Any], global_expl_vars: list[str],
                     config: dict[str, Any], model_config: dict[str, Any],
                     table_names: list[str]) -> Any:
    """Narrow one sub-model's candidate variables down with forward stepwise selection.

    `fit` fits one candidate specification of the sub-model, which is offered the global
    candidates plus the additional ones its family or its own entry declares. The candidates
    are checked against the tables the sub-model reads (`table_names`) first, so the run
    fails before any fitting if one of them is missing or empty. The stepwise defaults can
    be overridden with an optional 'stepwise' block in the family's estimate.yaml entry
    (max_vars, min_improvement); a model entry can carry an 'expected_sign' map
    (variable -> +1/-1) to rule out wrong-signed candidates.
    """
    candidates = _candidate_variables(global_expl_vars, config, model_config)
    if not candidates:
        raise ValueError(
            f"{model_config['name']}: no candidate variables to select from; check "
            "estimate.yaml's global_expl_vars and any addl_expl_vars entries."
        )

    settings = config.get("stepwise") or {}
    unknown = set(settings) - {"max_vars", "min_improvement"}
    if unknown:
        raise ValueError(
            f"{model_config['name']}: estimate.yaml's stepwise block has unknown keys "
            f"{sorted(unknown)}; only max_vars and min_improvement are supported."
        )
    expected_sign = model_config.get("expected_sign")
    for variable, sign in (expected_sign or {}).items():
        if sign not in (1, -1):
            raise ValueError(
                f"{model_config['name']}: expected_sign[{variable!r}] is {sign!r}, but "
                "expected signs must be 1 or -1."
            )

    _check_candidates(candidates, table_names, model_config)

    overrides = {key: settings[key] for key in ("max_vars", "min_improvement") if key in settings}
    history: list[dict[str, Any]] = []
    selected, m, bic = stepwise(
        fit, candidates, expected_sign=expected_sign, log=history, **overrides,
    )
    if m is None:
        raise ValueError(
            f"{model_config['name']}: stepwise selection kept no variables; check the "
            "candidate list and any expected_sign entries in estimate.yaml."
        )

    print(
        f"{model_config['name']}: stepwise kept {len(selected)} of {len(candidates)} "
        f"candidate variables (BIC {bic:,.1f})"
    )
    for entry in history:
        print("  + {added:<62} BIC {bic:>12,.1f}  pseudo-R2 {rho_bar_squared:.3f}".format(**entry))
    print("  selected: " + " + ".join(selected))
    return m


@orca.step("estimate_hlcm")
def estimate_hlcm(project_dir: Path) -> None:
    """Narrow each HLCM sub-model listed in estimate.yaml down with stepwise selection and register it."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "hlcm")
    global_expl_vars = _load_global_expl_vars(project_dir)
    base_year, historic_year = _load_calibration_period(project_dir)
    table_names = to_list(config["choosers"]) + to_list(config["alternatives"])

    for model_config in config["models"]:
        fit = partial(_fit_submodel, config, model_config, base_year, historic_year)
        # mm.register overwrites any previously registered model of the same name
        mm.register(_select_submodel(fit, global_expl_vars, config, model_config, table_names))


@orca.step("estimate_jlcm")
def estimate_jlcm(project_dir: Path) -> None:
    """Narrow each JLCM sub-model listed in estimate.yaml down with stepwise selection and register it."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "jlcm")
    global_expl_vars = _load_global_expl_vars(project_dir)
    base_year, historic_year = _load_calibration_period(project_dir)
    table_names = to_list(config["choosers"]) + to_list(config["alternatives"])

    for model_config in config["models"]:
        fit = partial(_fit_submodel, config, model_config, base_year, historic_year)
        # mm.register overwrites any previously registered model of the same name
        mm.register(_select_submodel(fit, global_expl_vars, config, model_config, table_names))


@orca.step("estimate_hulcm")
def estimate_hulcm(project_dir: Path) -> None:
    """Narrow each HULCM sub-model listed in estimate.yaml down with stepwise selection and register it."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "hulcm")
    global_expl_vars = _load_global_expl_vars(project_dir)
    base_year, historic_year = _load_calibration_period(project_dir)
    table_names = to_list(config["choosers"]) + to_list(config["alternatives"])

    for model_config in config["models"]:
        fit = partial(_fit_submodel, config, model_config, base_year, historic_year)
        # mm.register overwrites any previously registered model of the same name
        mm.register(_select_submodel(fit, global_expl_vars, config, model_config, table_names))


@orca.step("estimate_hupm")
def estimate_hupm(project_dir: Path) -> None:
    """Narrow each HUPM sub-model listed in estimate.yaml down with stepwise selection and register it."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "hupm")
    global_expl_vars = _load_global_expl_vars(project_dir)
    table_names = to_list(config["tables"])

    for model_config in config["models"]:
        fit = partial(_fit_hupm_submodel, config, model_config)
        # mm.register overwrites any previously registered model of the same name
        mm.register(_select_submodel(fit, global_expl_vars, config, model_config, table_names))


def add_run_args(parser):
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )

def run(args):
    """Run the orca steps listed under estimation_steps in estimate.yaml."""
    project_dir = Path(args.configs_dir).parent
    estimation_steps = _load_estimate_yaml(project_dir)["estimation_steps"]
    orca.add_injectable("project_dir", project_dir)
    orca.run(estimation_steps)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))