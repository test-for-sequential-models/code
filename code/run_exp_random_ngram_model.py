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

from .kernels import HammingKernel
from .kernels import ContiguousSubsequenceKernel
from .models import RandomNgramModel
from .utils import timer, with_spawn, sequence_to_seed
from .stein_operators import ZanellaSteinOperator
from .hypothesis_tests import SteinTestParametricBootstrap
from .hypothesis_tests import MMDTestParametricBootstrap
from .experiments import run_test
from .experiments import HeldTest
from .actions import SubActionSet
from .actions import PreActionSet
from .actions import MaxProbSubActionSet


def get_point(*, seed: int, model_a, model_b, perturbation_factor, N_bootstrap, desired_level,
              N_sequences,
              N_repeats_per_test, N_indep_tests):
    target_model = model_a
    sample_model = model_a.interpolate(model_b, 1 - perturbation_factor, perturbation_factor)
    n_states = len(model_a.alphabet)
    kernel = 'Hamming', HammingKernel()
    test_kwargs = dict(n_bootstrap=N_bootstrap, desired_level=desired_level, sample_size=N_sequences)
    hold = lambda con: lambda *args, **kwargs: HeldTest(lambda rng: con(*args, rng=rng, **kwargs))
    s_mmd, s_mmd2, s_stein = np.random.SeedSequence(seed).spawn(3)
    tests = {
        # f'MMD(k={kernel[0]}, n=100)': hold(MMDTestParametricBootstrap)(np.random.default_rng(s_mmd), kernel[1],
        #                                                                n_mmd_samples=100,
        #                                                                target_model=target_model, **test_kwargs),
        'MMD(k=CSK, n=100)': hold(MMDTestParametricBootstrap)(
            kernel=ContiguousSubsequenceKernel(min_length=3, max_length=3),
            n_mmd_samples=100,
            target_model=target_model,
            **test_kwargs
        )
        # f'MMD(k={kernel[0]}, n=same)': hold(MMDTestParametricBootstrap)(np.random.default_rng(s_mmd), kernel[1],
        #                                                                 n_mmd_samples=N_sequences,
        #                                                                 target_model=target_model, **test_kwargs),
    }
    stein_ops = {
        r"$\mathrm{ZS}_3'$": ZanellaSteinOperator(
            model=target_model,
            action_set=SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet),
            kappa='barker',
        ),
        r"$\mathrm{ZS}_{3,mp5}'$": ZanellaSteinOperator(
            model=target_model,
            action_set=(
                    SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
                    | MaxProbSubActionSet(model=target_model, tail=True, min_num_to_sub=5, max_num_to_sub=5)
            ),
            kappa='barker',
        ),
        r"$\mathrm{ZS}_{3,pre8}'$": ZanellaSteinOperator(
            model=target_model,
            action_set=(
                    SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
                    | PreActionSet(complement=False, min_ix=1, max_ix=8)
            ),
            kappa='barker',
        ),
    }
    # tests |= {
    #     f'Stein(k={kernel[0]}, op={k})': immediate(SteinTestParametricBootstrap)(kernel[1], v,
    #                                                                              rng=np.random.default_rng(s),
    #                                                                              target_model=target_model,
    #                                                                              **test_kwargs)
    #     for (k, v), s in with_spawn(stein_ops.items(), s_stein)
    # }
    tests |= {
        f'Stein(k=CSK, op={k})': hold(SteinTestParametricBootstrap)(
            kernel=ContiguousSubsequenceKernel(min_length=3, max_length=3),
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
    vars_to_plot = ['rejection_rate']
    # vars_to_plot = ['total_seconds']
    for var_to_plot in vars_to_plot:
        for parameter, values in results.items():
            to_plot = {k: {k2: v2[var_to_plot] for k2, v2 in v.items()} for k, v in values.items()}
            df = pd.DataFrame(to_plot).T
            df.index.name = parameter
            df.pipe(lambda x: sns.lineplot(data=x))
            plt.title(var_to_plot)
            if var_to_plot == 'rejection_rate':
                plt.gca().set_ylim([-.02, 1.02])
                plt.gca().axhline(0.05, ls="dotted", c='gray')
            if parameter in {'N_sequences', 'concentration_param'}:
                plt.gca().set_xscale('log')
            plt.show()


def run_experiment(*, test_run):
    baseline_parameters = {
        'N_sequences': 50,
        'desired_level': 0.05,
        'perturbation_factor': 1,
        # 'E_length': 8,
        'E_length': 20,
        'n_in_ngram': 2,
        'n_states': 10,
        'concentration_param': 1,
    }
    if test_run:
        baseline_parameters |= {
            'N_indep_tests': 1,
            'N_repeats_per_test': 1,
            'N_bootstrap': 2,
        }
    else:
        baseline_parameters |= {
            'N_indep_tests': 1,
            'N_repeats_per_test': 10,
            'N_bootstrap': 100,
        }
    if test_run:
        parameters_to_vary = {
            'N_sequences': [6],
        }
    else:
        parameters_to_vary = {
            # 'N_sequences': np.logspace(1, 2, 5, dtype=int),
            'perturbation_factor': np.linspace(0, 1, 5),
            # 'E_length': np.linspace(2, 80, 10),
            # 'concentration_param': np.logspace(-2, 2, 5),
        }

    rng_seed = 305566025966302217582007613457990816426
    s = np.random.SeedSequence(rng_seed)
    results = dict()

    def go():
        s_model, s_experiment = s.spawn(2)
        model_seed = sequence_to_seed(s_model)
        jobs = []

        def run_job(parameter, value, s_value):
            # we derive all pairs of models from the same seed,
            # so that when the parameters are the same we get the same model.
            # this helps to reduce noise.
            # alternatively we could just average multiple runs of random models.
            # perhaps in the future we will do this.
            model_rng = np.random.default_rng(model_seed)
            model_param_keys = {'E_length', 'n_in_ngram', 'n_states', 'concentration_param'}
            varied_params = baseline_parameters | {parameter: value}
            model_params = {k: varied_params[k] for k in model_param_keys}
            model_a = RandomNgramModel(rng=model_rng, **model_params)
            model_b = RandomNgramModel(rng=model_rng, **model_params)
            point_params = {k: varied_params[k] for k in varied_params.keys() - model_param_keys}
            point_params['model_a'] = model_a
            point_params['model_b'] = model_b
            if not test_run:
                print(parameter, value)
            s_point, s_tests = s_value.spawn(2)
            point = get_point(seed=sequence_to_seed(s_point), **point_params)
            retval = dict()
            for (test_name, test), s_test in with_spawn(point['tests'].items(), s_tests):
                test_kwargs = point | {'test': test}
                del test_kwargs['tests']
                retval[test_name] = run_test(seed=sequence_to_seed(s_test), **test_kwargs)
            return retval

        for (parameter, values), s_param in with_spawn(parameters_to_vary.items(), s_experiment):
            results[parameter] = dict()
            for value, s_value in with_spawn(values, s_param):
                job = parameter, value, s_value
                jobs.append(job)
        job_results = map(run_job, *zip(*jobs))
        for (parameter, value, s_value), job_result in zip(jobs, job_results):
            results[parameter][value] = job_result

    if test_run:
        go()
    else:
        with timer():
            go()
    return results


experiment_name = 'new_exp_infra_random_ngram_model'


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
        raise Exception()
        results = load_results()
    except:
        results = run_experiment(test_run=False)
        save_results(results)
    plot_results(results)


if __name__ == '__main__':
    main()
