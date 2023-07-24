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
from .models import SimpleMRFModel
from .models import NgramModel
from .kernels import ContiguousSubsequenceKernel
from .stein_operators import ZanellaSteinOperator
from .hypothesis_tests import MMDTestWildBootstrap
from .hypothesis_tests import SteinTestWildBootstrap
from .hypothesis_tests import LikelihoodRatioTest
from .actions import SubActionSet
from .actions import InsertActionSet
from .actions import DelActionSet
from .actions import Composition


def get_point(
        *,
        N_sequences: int,
        N_repeats_per_test: int,
        N_indep_tests: int,
        N_bootstrap: int,
        desired_level: float,
        markov_order: int,
        max_length: int,
        target_concentration_param: float,
        sample_concentration_param: float,
        n_states: int,
        length_potential: float,
):
    target_model = SimpleMRFModel(
        length_potential=length_potential,
        n_states=n_states,
        concentration_param=target_concentration_param,
        max_length=max_length,
    )
    sample_model = SimpleMRFModel(
        length_potential=length_potential,
        n_states=n_states,
        concentration_param=sample_concentration_param,
        max_length=max_length,
    )
    test_kwargs = dict(n_bootstrap=N_bootstrap, desired_level=desired_level, sample_size=N_sequences)
    hold = lambda fn: lambda *args, **kwargs: HeldTest(lambda rng: fn(*args, rng=rng, **kwargs))

    stein_ops = {}
    for label, max_ix in (('infty', None),):
        stein_ops |= {
            r"$\mathrm{ZS}_{" + label + ",mpf}$": ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        SubActionSet(complement=True, min_ix=0, max_ix=max_ix, xs=target_model.alphabet)
                        | InsertActionSet(complement=True, min_ix=0, max_ix=max_ix, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=max_ix)
                ),
                kappa='mpf',
            ),
            r"$\mathrm{ZS}_{" + label + ",b}$": ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        SubActionSet(complement=True, min_ix=0, max_ix=max_ix, xs=target_model.alphabet)
                        | InsertActionSet(complement=True, min_ix=0, max_ix=max_ix, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=max_ix)
                ),
                kappa='barker',
            ),
        }
    stein_ops |= {
        r"$\mathrm{ZS}_{d=2,b}$": ZanellaSteinOperator(
            model=target_model,
            action_set=(
                    SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                    | InsertActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                    | DelActionSet(complement=True, min_ix=0, max_ix=None)
                    | Composition(
                        SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet),
                        SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet),
                    )
            ),
            kappa='barker',
        ),
    }

    tests = {}
    for kernel in (
            ('CSK', ContiguousSubsequenceKernel(min_length=markov_order + 1, max_length=markov_order + 1)),
    ):
        tests |= {
            f'MMD-Wild(k={kernel[0]}, n=same)': hold(MMDTestWildBootstrap)(
                kernel=kernel[1],
                n_mmd_samples=N_sequences,
                target_model=target_model,
                **test_kwargs
            ),
        }
        tests |= {
            f'Stein-Wild(k={kernel[0]}, op={k})': hold(SteinTestWildBootstrap)(
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
            **(test_kwargs | {'n_bootstrap': 100}),
        ),
        r'LR(oracle)': hold(LikelihoodRatioTest)(
            target_model=target_model,
            get_mle_model=lambda xs: sample_model,
            **(test_kwargs | {'n_bootstrap': 100}),
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
            plt.show()


experiment_name = 'exp_simple_mrf'


def run_experiment(*, test_run):
    version = '6'
    baseline_parameters = {
        'N_sequences': 200,
        'desired_level': 0.05,
        'target_concentration_param': 1,
        'sample_concentration_param': 0.9,
        'markov_order': 1,
        'max_length': 20,
        'n_states': 3,
        'length_potential': 0,
    }
    if test_run:
        baseline_parameters |= {
            'N_indep_tests': 1,
            'N_repeats_per_test': 1,
            'N_bootstrap': 10,
        }
    else:
        baseline_parameters |= {
            # 'N_indep_tests': 10,
            # 'N_repeats_per_test': 10,
            # 'N_indep_tests': 4,
            # 'N_repeats_per_test': 40,
            'N_indep_tests': 20,
            'N_repeats_per_test': 50,
            'N_bootstrap': 1_000,
        }
    if test_run:
        parameters_to_vary = {
            'N_sequences': [1],
        }
    else:
        parameters_to_vary = {
            # 'N_sequences': np.linspace(200, 700, 6, dtype=int),
            # 'sample_concentration_param': np.linspace(0.5, 1.5, 9),
            'sample_concentration_param': np.linspace(0.75, 1.25, 9),
            # 'sample_concentration_param': np.linspace(1, 1, 1),
        }

    root_seed = 9824740821212930092259595777667619796, ''
    results = dict()

    from random import sample
    def go():
        for parameter, values in sample(list(parameters_to_vary.items()), k=len(parameters_to_vary)):
            results[parameter] = dict()
            for value in sample(list(values), k=len(values)):
                if not test_run:
                    print(parameter, value, values)
                results[parameter][value] = dict()
                point = get_point(**(baseline_parameters | {parameter: value}))
                for test_name, test in point['tests'].items():
                    if not test_run:
                        print('', test_name)
                    test_kwargs = point | {'test': test, 'test_run': test_run}
                    del test_kwargs['tests']
                    value_part = slugify(str(value))
                    cache_path = f'../figures/{experiment_name}/v{version}__{slugify(parameter)}__{value_part}__{slugify(test_name)}.pickle'
                    results[parameter][value][test_name] = the_results = run_test(
                        seed=extend_seed(root_seed, f'{parameter}/{value}/{test_name}'),
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
