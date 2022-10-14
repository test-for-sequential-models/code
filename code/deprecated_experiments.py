import pandas as pd
from collections import defaultdict
from typing import Callable
from typing import Optional
from typing import Iterable
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from dataclasses import fields as dc_fields

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from distributed import Client

from .free_vector_space import FreeVector
from .stein_operators import SteinOperator
from .kernels import Kernel
from .kernels import SteinKernel
from .models import SequenceDist
from .hypothesis_tests import compute_U_statistics, compute_wild_bootstrap_kernels
from .hypothesis_tests import compute_gram_matrix
from .hypothesis_tests import generate_aux_wild_bootstrap_processes


@dataclass
class ExperimentPoint:
    kernels: dict[str, Kernel]
    stein_operators: dict[str, SteinOperator]
    target_model: SequenceDist
    sample_model: SequenceDist
    N_p: int  # this should always be == N_q, for now
    N_q: int
    N_bootstrap_D: int
    N_bootstrap_MMD: int
    N_bootstrap_wild: int
    do_wild_bootstrap: bool
    verify_wild_bootstrap: bool

    def __hash__(self):
        ident = lambda x: x
        special_ops = defaultdict(lambda: ident) | {
            'kernels': tuple,
            'stein_operators': tuple,
        }
        get_field_hashable = lambda name: special_ops[name](getattr(self, name))
        tuple_for_hash = tuple(get_field_hashable(field.name) for field in dc_fields(self))
        return hash(tuple_for_hash)


@dataclass
class SingleVarResults:
    test_error_rates: pd.DataFrame


def compute_MMD_kernel(kernel: Kernel, X: list[FreeVector], Y: list[FreeVector]) -> np.ndarray:
    N_p = len(X)
    N_q = len(Y)
    assert N_p == N_q, "unimplemented"
    KXX = kernel.compute_gram_single(X)
    KXY = kernel.compute_gram(X, Y)
    KYY = kernel.compute_gram_single(Y)
    H = KXX + KYY - KXY - KXY.T
    assert not np.isnan(KXX).any(), "KXX"
    assert not np.isnan(KYY).any(), "KYY"
    assert not np.isnan(KXY).any(), "KXY"
    assert not np.isnan(H).any(), "H"
    return H


def compute_samples(experiment_point: ExperimentPoint, rng, client: Optional[Client]) -> dict[str, dict[str, dict]]:
    get_samples = lambda model, rng, N: [
        FreeVector(tuple(s), s) for s in model.generate_samples(rng, N)
    ]
    result = {}

    def random_samples(rng, N, f):
        seeds = rng.bit_generator.random_raw(N)
        if client is not None:
            result = client.map(lambda seed: f(np.random.default_rng(seed)), seeds)
            return client.submit(np.stack, result).result()
        else:
            result = list(map(lambda seed: f(np.random.default_rng(seed)), seeds))
            return np.stack(result)

    for kernel_name, kernel in experiment_point.kernels.items():
        result[kernel_name] = {'by_stein_operator': {}}
        for stein_op_name, stein_operator in experiment_point.stein_operators.items():
            stein_kernel_samples = random_samples(
                rng, experiment_point.N_bootstrap_D,
                lambda rng: compute_gram_matrix(
                    SteinKernel(kernel, stein_operator),
                    get_samples(experiment_point.sample_model, rng, experiment_point.N_q),
                )
            )
            result[kernel_name]['by_stein_operator'][stein_op_name] = stein_kernel_samples
        mmd_kernel_samples = random_samples(
            rng, experiment_point.N_bootstrap_MMD,
            lambda rng: compute_MMD_kernel(
                kernel,
                get_samples(experiment_point.target_model, rng, experiment_point.N_p),
                get_samples(experiment_point.sample_model, rng, experiment_point.N_q),
            )
        )
        result[kernel_name]['mmd_kernel'] = dict(mmd_kernel=mmd_kernel_samples)
    return result


def compute_test_error_rates(
        experiment_point: ExperimentPoint,
        rng,
        client: Optional[Client],
        bootstrap_cache: Optional[dict[any, any]] = None,
):
    experiment_point_h1 = experiment_point
    experiment_point_h0 = dc_replace(experiment_point_h1, sample_model=experiment_point.target_model)

    def map_sample_dict(f, sample_dict):
        return {
            k1: {
                k2: {
                    k3: f(v3)
                    for k3, v3 in v2.items()
                }
                for k2, v2 in v1.items()
            }
            for k1, v1 in sample_dict.items()
        }

    alt_hyp_K_samples = compute_samples(experiment_point_h1, rng, client)
    alt_hyp_samples = map_sample_dict(compute_U_statistics, alt_hyp_K_samples)
    want_param_bootstrap = experiment_point.verify_wild_bootstrap or not experiment_point.do_wild_bootstrap
    if want_param_bootstrap:
        get_bootstrap_samples = lambda: compute_samples(experiment_point_h0, rng, client)
        if bootstrap_cache is not None:
            if experiment_point_h0 not in bootstrap_cache:
                bootstrap_cache[experiment_point_h0] = get_bootstrap_samples()
            null_hyp_K_samples = bootstrap_cache[experiment_point_h0]
        else:
            null_hyp_K_samples = get_bootstrap_samples()
        null_hyp_samples = map_sample_dict(compute_U_statistics, null_hyp_K_samples)
        if not experiment_point.do_wild_bootstrap:
            bootstrap_samples = null_hyp_samples
    if experiment_point.do_wild_bootstrap:
        assert experiment_point.N_q == experiment_point.N_p, "unimplemented"
        Z = generate_aux_wild_bootstrap_processes(rng, experiment_point.N_q, experiment_point.N_bootstrap_wild)
        bootstrap_samples = map_sample_dict(lambda K: compute_U_statistics(compute_wild_bootstrap_kernels(K, Z)),
                                            alt_hyp_K_samples)
        if experiment_point.verify_wild_bootstrap:
            Z = generate_aux_wild_bootstrap_processes(rng, experiment_point.N_q, experiment_point.N_bootstrap_wild)
            verification_samples = map_sample_dict(lambda K: compute_U_statistics(compute_wild_bootstrap_kernels(K, Z)),
                                                   null_hyp_K_samples)
    result = {}
    for kernel_name in experiment_point.kernels.keys():
        get = lambda samples: samples[kernel_name]['mmd_kernel']['mmd_kernel']
        crit_val = lambda samples: np.quantile(get(samples), 0.95, axis=-1)
        mmd_type_II_error = (get(alt_hyp_samples) <= crit_val(bootstrap_samples)).mean(axis=-1).item()
        result |= {
            f'MMD type II error (k={kernel_name})': mmd_type_II_error,
        }
        if experiment_point.verify_wild_bootstrap:
            mmd_type_I_error = (get(null_hyp_samples) > crit_val(verification_samples)).mean(axis=-1).item()
            result |= {
                f'MMD type I error (k={kernel_name})': mmd_type_I_error,
            }
        for stein_op_name in experiment_point.stein_operators.keys():
            get = lambda samples: samples[kernel_name]['by_stein_operator'][stein_op_name]
            crit_val = lambda samples: np.quantile(get(samples), 0.95, axis=-1)
            stein_type_II_error = (get(alt_hyp_samples) <= crit_val(bootstrap_samples)).mean(axis=-1).item()
            result |= {
                f'Stein type II error (k={kernel_name}, op={stein_op_name})': stein_type_II_error,
            }
            if experiment_point.verify_wild_bootstrap:
                stein_type_I_error = (get(null_hyp_samples) > crit_val(verification_samples)).mean(axis=-1).item()
                result |= {
                    f'Stein type I error (k={kernel_name}, op={stein_op_name})': stein_type_I_error,
                }
    return result


@dataclass
class Experiment:
    constructor: Callable[[dict[str, any]], ExperimentPoint]
    baseline_parameters: dict[str, any]
    parameters_to_vary: dict[str, Iterable[any]]

    def run(
            self,
            rng: np.random.Generator,
            *,
            variables: Optional[list[str]] = None,
            client: Optional[Client] = None,
    ) -> dict[str, SingleVarResults]:
        output = {}
        bootstrap_cache = {}
        items = self.parameters_to_vary.items()
        if variables is not None:
            items = [(var, self.parameters_to_vary[var]) for var in variables]
        for variable, values in items:
            params = self.baseline_parameters.copy()
            test_error_rates = {}
            for value in values:
                print(variable, value)  # TODO do some smart tqdm thing here
                params[variable] = value
                point = self.constructor(params)
                test_error_rates[value] = compute_test_error_rates(point, rng, client, bootstrap_cache)
            results = SingleVarResults(
                test_error_rates=pd.DataFrame(test_error_rates).T,
            )
            results.test_error_rates.index.name = variable
            output[variable] = results
        return output


def plot_output(
        output: dict[str, SingleVarResults],
        *,
        experiment_name: Optional[str] = None,
        log_scale: Optional[set[str]] = None,
        plot_filter_regex: Optional[str] = None,
):
    for variable, results in output.items():
        if plot_filter_regex is not None:
            data = results.test_error_rates.filter(regex=plot_filter_regex)
        else:
            data = results.test_error_rates
        sns.lineplot(data=data)
        if log_scale is not None and variable in log_scale:
            plt.gca().set_xscale('log')
        plt.gca().set_ylim([-.02, 1.02])
        plt.title(f'Experiment: {experiment_name}')
        plt.gca().axhline(0.05, label=r'$\alpha$', color='gray', ls='--')
        plt.gca().axhline(0.95, label=r'$1-\alpha$', color='gray', ls='--')
        if experiment_name is not None:
            plt.savefig(f'../figures/{experiment_name}_error_rates_by_{variable}.pdf')
        plt.show()
