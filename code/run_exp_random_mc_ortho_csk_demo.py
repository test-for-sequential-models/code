# fix relative imports
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    __package__ = Path(__file__).parent.name
# end of relative imports fix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .experiments import run_test
from .experiments import HeldTest
from .utils import timer, extend_seed
from .utils import slugify
from .utils import get_rng
from .models import NgramModel
from .models import RandomNgramModel
from .kernels import ContiguousSubsequenceKernel
from .kernels import ProductKernel
from .kernels import DiracKernelOfLength
from .stein_operators import ZanellaSteinOperator
from .hypothesis_tests import MMDTestParametricBootstrap
from .hypothesis_tests import SteinTestParametricBootstrap
from .hypothesis_tests import SteinTestWildBootstrap
from .hypothesis_tests import LikelihoodRatioTest
from .actions import SubActionSet
from .actions import InsertActionSet
from .actions import DelActionSet


def get_point(
        *,
        seed,
        nth_call: int,
        N_sequences: int,
        N_repeats_per_test: int,
        N_indep_tests: int,
        N_bootstrap: int,
        desired_level: float,
        markov_order: int,
        perturbation: float,
        n_states: int,
        E_length: float,
):
    assert 0 <= perturbation <= 1
    target_model = RandomNgramModel(
        rng=get_rng(seed, 'random-ngram', 3, nth_usage=nth_call + 1),
        n_in_ngram=1,
        n_states=n_states,
        E_length=E_length,
        concentration_param=1,
    )
    sample_model = RandomNgramModel(
        rng=get_rng(seed, 'random-ngram', 4, nth_usage=nth_call + 1),
        n_in_ngram=1,
        n_states=n_states,
        E_length=E_length,
        concentration_param=1,
    )
    sample_model = sample_model.interpolate(target_model, perturbation, 1 - perturbation)
    test_kwargs = dict(n_bootstrap=N_bootstrap, desired_level=desired_level, sample_size=N_sequences)
    hold = lambda fn: lambda *args, **kwargs: HeldTest(lambda rng: fn(*args, rng=rng, **kwargs))

    stein_ops = {}
    for label, max_ix in (('infty', None),):
        stein_ops |= {
            r"$\mathrm{ZS}_{" + label + ",b}'$": ZanellaSteinOperator(
                model=target_model,
                action_set=(
                    SubActionSet(complement=True, min_ix=0, max_ix=max_ix, xs=target_model.alphabet)
                ),
                kappa='barker',
            ),
        }

    tests = {}
    csk_kernel = ContiguousSubsequenceKernel(min_length=markov_order + 1, max_length=markov_order + 1)
    for kernel in (
            ('CSK', csk_kernel),
            ('Ortho-CSK', ProductKernel(DiracKernelOfLength(), csk_kernel)),
    ):
        tests |= {
            f'MMD(k={kernel[0]}, n=100)': hold(MMDTestParametricBootstrap)(
                kernel=kernel[1],
                n_mmd_samples=100,
                target_model=target_model,
                **test_kwargs
            ),
        }
        tests |= {
            f'Stein(k={kernel[0]}, op={k})': hold(SteinTestParametricBootstrap)(
                kernel=kernel[1],
                stein_op=v,
                target_model=target_model,
                **test_kwargs
            )
            for k, v in stein_ops.items()
        }

    tests |= {
        r'LR($k=k_\mathrm{model}$)': hold(LikelihoodRatioTest)(
            target_model=target_model,
            get_mle_model=lambda xs: NgramModel.fit(n_in_ngram=markov_order, n_states=n_states, sequences=xs),
            **test_kwargs,
        ),
        r'LR(oracle)': hold(LikelihoodRatioTest)(
            target_model=target_model,
            get_mle_model=lambda xs: sample_model,
            **test_kwargs,
        ),
    }
    point = dict(
        tests=tests,
        sample_model=sample_model,
        N_sequences=N_sequences,
        N_repeats_per_test=N_repeats_per_test,
        N_indep_tests=N_indep_tests,
    )
    return point


def plot_results(results):
    for var_to_plot in ['rejection_rate']:
        for parameter, values in results.items():
            to_plot = {k: {k2: v2[var_to_plot] for k2, v2 in v.items()} for k, v in values.items()}
            df = pd.DataFrame(to_plot).T
            df.index.name = parameter
            filename = f'../figures/{experiment_name}_{var_to_plot}_by_{parameter}.csv'
            df_to_write = df.copy()
            safeify = lambda x: x.replace(',', ':').replace('$', '_').replace('\\', '_')
            df_to_write.columns = df_to_write.columns.map(lambda x: '{' + safeify(x) + '}')
            df_to_write.to_csv(filename)
            df.pipe(lambda x: sns.lineplot(data=x))
            plt.title(var_to_plot)
            if var_to_plot == 'rejection_rate':
                plt.gca().set_ylim([-.02, 1.02])
                plt.gca().axhline(0.05, ls="dotted", c='gray')
            if parameter == 'N_sequences':
                plt.gca().set_xscale('log')
            plt.show()


experiment_name = 'exp_random_mc_ortho_csk_demo'


def run_experiment(*, test_run):
    version = '1'
    baseline_parameters = {
        'N_sequences': 30,
        'desired_level': 0.05,
        'perturbation': 0.5,
        'markov_order': 2,
        'n_states': 10,
        'E_length': 8,
    }
    if test_run:
        baseline_parameters |= {
            'N_indep_tests': 1,
            'N_repeats_per_test': 1,
            'N_bootstrap': 10,
        }
    else:
        baseline_parameters |= {
            'N_indep_tests': 4,
            'N_repeats_per_test': 40,
            # 'N_indep_tests': 10,
            # 'N_repeats_per_test': 100,
            'N_bootstrap': 100,
        }
    if test_run:
        parameters_to_vary = {
            'N_sequences': [1],
        }
    else:
        parameters_to_vary = {
            # 'N_sequences': np.logspace(1, 2.5, 5, dtype=int),
            # 'perturbation': np.linspace(0, 1, 3),
            'E_length': np.linspace(2, 20, 5),
        }

    root_seed = 60270677365435319306569075866992992285, ''
    results = dict()

    def go():
        for parameter, values in parameters_to_vary.items():
            results[parameter] = dict()
            for i, value in enumerate(values):
                if not test_run:
                    print(parameter, value, values)
                results[parameter][value] = dict()
                point = get_point(
                    seed=extend_seed(root_seed, f'{parameter}/for-get_point'),
                    nth_call=i,
                    **(baseline_parameters | {parameter: value}),
                )
                for test_name, test in point['tests'].items():
                    if not test_run:
                        print('', test_name)
                    test_kwargs = point | {'test': test, 'test_run': test_run}
                    del test_kwargs['tests']
                    value_part = slugify(str(value))
                    cache_path = f'../figures/{experiment_name}/v{version}__{slugify(parameter)}__{value_part}__{slugify(test_name)}.pickle'
                    results[parameter][value][test_name] = the_results = run_test(
                        seed=extend_seed(root_seed, f'{parameter}/{value}/{test_name}/for-run_test'),
                        cache_path=cache_path,
                        **test_kwargs,
                    )
                    if not test_run:
                        print(' ', the_results['rejection_rate'])

    if test_run:
        go()
    else:
        with timer():
            go()
    return results


def main():
    results = run_experiment(test_run=False)
    plot_results(results)


if __name__ == '__main__':
    main()
