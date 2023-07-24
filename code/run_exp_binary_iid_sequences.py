# fix relative imports
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    __package__ = Path(__file__).parent.name
# end of relative imports fix

import json
import datetime
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .experiments import run_test
from .experiments import HeldTest
from .utils import timer, with_spawn, sequence_to_seed
from .models import PoissonBernoulliModel
from .kernels import HammingKernel
from .stein_operators import GibbsSteinOperator
from .stein_operators import OldStyleZanellaSteinOperator
from .hypothesis_tests import MMDTestParametricBootstrap
from .hypothesis_tests import SteinTestParametricBootstrap


def get_point(*, seed: int, N_sequences, N_repeats_per_test, N_indep_tests, lmbda, sample_p, N_bootstrap,
              desired_level):
    target_model = PoissonBernoulliModel(lmbda, 0.5)
    sample_model = PoissonBernoulliModel(lmbda, sample_p)
    kernel = 'Hamming', HammingKernel()
    test_kwargs = dict(n_bootstrap=N_bootstrap, desired_level=desired_level, sample_size=N_sequences)
    hold = lambda fn: lambda *args, **kwargs: HeldTest(lambda rng: fn(*args, rng=rng, **kwargs))

    tests = {
        f'MMD(k={kernel[0]}, n=100)': hold(MMDTestParametricBootstrap)(
            kernel=kernel[1],
            n_mmd_samples=100,
            target_model=target_model,
            **test_kwargs
        ),
        f'MMD(k={kernel[0]}, n=same)': hold(MMDTestParametricBootstrap)(
            kernel=kernel[1],
            n_mmd_samples=N_sequences,
            target_model=target_model,
            **test_kwargs
        ),
    }
    stein_ops = {
        r'$\mathrm{ZS}_1$': OldStyleZanellaSteinOperator(target_model, ['insert', 'replace', 'pop'], 1),
        r'$\mathrm{ZS}_\infty$': OldStyleZanellaSteinOperator(target_model, ['insert', 'replace', 'pop'], None),
        r'$\mathrm{ZS}_{\infty,dt}$': OldStyleZanellaSteinOperator(target_model,
                                                                   ['insert', 'replace', 'pop', 'drop_tail'],
                                                                   None),
        r"$\mathrm{ZS}_\infty'$": OldStyleZanellaSteinOperator(target_model, ['replace'], None),
        'Gibbs': GibbsSteinOperator(target_model),
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


def run_experiment(*, test_run):
    baseline_parameters = {
        'N_sequences': 50,
        'sample_p': 0.3,
        'lmbda': 10,
        'desired_level': 0.05,
    }
    if test_run:
        baseline_parameters |= {
            'N_indep_tests': 1,
            'N_repeats_per_test': 1,
            'N_bootstrap': 10,
        }
    else:
        baseline_parameters |= {
            'N_indep_tests': 10,
            'N_repeats_per_test': 10,
            'N_bootstrap': 100,
        }
    if test_run:
        parameters_to_vary = {
            'N_sequences': [1],
        }
    else:
        parameters_to_vary = {
            'N_sequences': np.logspace(1, 2.5, 5, dtype=int),
            'sample_p': np.linspace(0.2, 0.45, 5),
            'lmbda': np.linspace(5, 40, 5),
        }

    rng_seed = 49790651075799856659082344803041995768
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


experiment_name = 'new_exp_infra_binary_iid_sequences'


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
