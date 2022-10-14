# fix relative imports
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    __package__ = Path(__file__).parent.name
# end of relative imports fix

import pickle
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .kernels import GaussianKernel
from .models import RandomWalk
from .models import RandomWalkWithHolding
from .stein_operators import ZanellaSteinOperator
from .utils import timer, with_spawn, sequence_to_seed
from .hypothesis_tests import SteinTestParametricBootstrap
from .hypothesis_tests import MMDTestParametricBootstrap
from .experiments import run_test
from .experiments import HeldTest
from .actions import SubActionSet
from .actions import MaxProbSubActionSet


def get_point(*, seed: int, E_length, N_sequences, N_repeats_per_test, N_indep_tests, n_states, holding_probability,
              n_holding_states, desired_level, N_bootstrap):
    target_model = RandomWalk(n_states=n_states, E_length=E_length)
    sample_model = RandomWalkWithHolding(
        n_states=n_states, E_length=E_length,
        holding_probability=holding_probability, n_holding_states=n_holding_states)
    kernel = 'Gaussian', GaussianKernel(n_states=n_states)
    test_kwargs = dict(n_bootstrap=N_bootstrap, desired_level=desired_level, sample_size=N_sequences)
    hold = lambda con: lambda *args, **kwargs: HeldTest(lambda rng: con(*args, rng=rng, **kwargs))
    s_mmd, s_stein = np.random.SeedSequence(seed).spawn(2)
    tests = {
        f'MMD(k={kernel[0]}, n=100)': hold(MMDTestParametricBootstrap)(
            kernel=kernel[1],
            n_mmd_samples=100,
            target_model=target_model,
            **test_kwargs
        ),
        # f'MMD(k={kernel[0]}, n=same)': hold(MMDTestParametricBootstrap)(np.random.default_rng(s_mmd), kernel[1],
        #                                                                 n_mmd_samples=N_sequences,
        #                                                                 target_model=target_model, **test_kwargs),
    }
    stein_ops = {
        r"$\mathrm{ZS}_5'$": ZanellaSteinOperator(
            model=target_model,
            action_set=SubActionSet(complement=True, min_ix=0, max_ix=4, xs=target_model.alphabet),
            kappa='barker',
        ),
        r"$\mathrm{ZS}_{5,mp}'$": ZanellaSteinOperator(
            model=target_model,
            action_set=(
                    SubActionSet(complement=True, min_ix=0, max_ix=4, xs=target_model.alphabet)
                    | MaxProbSubActionSet(model=target_model, tail=True, min_num_to_sub=1, max_num_to_sub=5)
            ),
            kappa='barker',
        ),
        # r"$\mathrm{ZS}_5$": OldStyleZanellaSteinOperator(target_model, ['replace', 'insert', 'pop'], 5),
        # r"$\mathrm{ZS}_{5,dt}'$": OldStyleZanellaSteinOperator(target_model, ['replace', 'drop_tail'], 5),
        # r"$\mathrm{ZS}_{5,dt}$": OldStyleZanellaSteinOperator(target_model, ['replace', 'insert', 'pop', 'drop_tail'], 5),
    }
    tests |= {
        f'Stein(k={kernel[0]}, op={k})': hold(SteinTestParametricBootstrap)(
            kernel=kernel[1],
            stein_op=v,
            target_model=target_model,
            **test_kwargs
        )
        for (k, v), s in with_spawn(stein_ops.items(), s_stein)
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
    for var_to_plot in ['rejection_rate', 'avg_total_seconds']:
        for parameter, values in results.items():
            to_plot = {k: {k2: v2[var_to_plot] for k2, v2 in v.items()} for k, v in values.items()}
            df = pd.DataFrame(to_plot).T
            df.index.name = parameter
            df.pipe(lambda x: sns.lineplot(data=x))
            plt.title(var_to_plot)
            if var_to_plot == 'rejection_rate':
                plt.gca().set_ylim([-.02, 1.02])
                plt.gca().axhline(0.05, ls="dotted", c='gray')
            if parameter == 'N_sequences':
                plt.gca().set_xscale('log')
            plt.show()


def run_experiment(*, test_run):
    baseline_parameters = {
        'N_sequences': 50,
        'n_states': 8,
        'E_length': 10,
        # 'holding_probability': 1 / 3,
        # 'n_holding_states': 3,
        'holding_probability': 1 / 2,
        'n_holding_states': 8,
        'desired_level': 0.05,
    }
    if test_run:
        baseline_parameters |= {
            'N_indep_tests': 1,
            'N_repeats_per_test': 1,
            'N_bootstrap': 2,
        }
    else:
        baseline_parameters |= {
            'N_indep_tests': 3,
            'N_repeats_per_test': 3,
            'N_bootstrap': 100,
        }
    if test_run:
        parameters_to_vary = {
            'N_sequences': [10],
        }
    else:
        parameters_to_vary = {
            # 'N_sequences': np.logspace(1, 2, 5, dtype=int),
            # 'n_states': np.linspace(2, 20, 5, dtype=int),
            # 'E_length': np.linspace(2, 100, 5),
            # 'n_holding_states': np.linspace(1, 8, 8, dtype=int),
            'holding_probability': np.linspace(0, 1, 5),
        }

    rng_seed = 31077066854702875869046709392664956469
    s = np.random.SeedSequence(rng_seed)
    results = dict()

    def go():
        for (parameter, values), s_param in with_spawn(parameters_to_vary.items(), s):
            results[parameter] = dict()
            for value, s_value in with_spawn(values, s_param):
                if not test_run:
                    print(parameter, value)
                results[parameter][value] = dict()
                s_point, s_tests = s_param.spawn(2)
                point = get_point(seed=sequence_to_seed(s_point), **(baseline_parameters | {parameter: value}))
                for (test_name, test), s_test in with_spawn(point['tests'].items(), s_tests):
                    test_kwargs = point | {'test': test}
                    del test_kwargs['tests']
                    results[parameter][value][test_name] = run_test(seed=sequence_to_seed(s_test), **test_kwargs)

    if test_run:
        go()
    else:
        with timer():
            go()
    return results


# what if the transition is extremely unlikely to be at the end of the sequence?
# that should make Z-S non-growing pretty bad? not 100% sure here, think more


experiment_name = 'new_exp_infra_markov_chain'


def save_results(results):
    for parameter, values in results.items():
        if type(list(values.keys())[0]) == np.int64:
            values = {
                int(k): v for k, v in values.items()
            }
            results[parameter] = values
    with open(f'../figures/{experiment_name}_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    with open(f'../figures/{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def load_results():
    with open(f'../figures/{experiment_name}_results.pickle', 'rb') as f:
        results = pickle.load(f)
    return results


def main():
    try:
        results = load_results()
    except:
        results = run_experiment(test_run=False)
        save_results(results)
    plot_results(results)


if __name__ == '__main__':
    main()
