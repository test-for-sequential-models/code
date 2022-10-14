# fix relative imports
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    __package__ = Path(__file__).parent.name
# end of relative imports fix

import numpy as np

from .deprecated_experiments import Experiment
from .deprecated_experiments import ExperimentPoint
from .deprecated_experiments import plot_output
from .kernels import HammingKernel
from .models import NgramModel
from .stein_operators import GibbsSteinOperator
from .stein_operators import OldStyleZanellaSteinOperator
from .utils import timer


def get_point(*, rng, alpha_target, alpha_sample, E_length, N_q, n_states):
    def construct_model(alpha):
        transitions = np.zeros(n_states + 1)
        transitions[1:] = 1 / np.arange(1, n_states + 1) ** alpha
        transitions[1:] /= transitions[1:].sum()
        transitions[0] = 1 / E_length
        transitions[1:] *= 1 - transitions[0]
        # print(transitions)
        return NgramModel(transitions)

    target_model = construct_model(alpha_target)
    sample_model = construct_model(alpha_sample)
    # print(target_model.variable_length_samples(rng, 4))
    # print(sample_model.variable_length_samples(rng, 4))
    return ExperimentPoint(
        target_model=target_model,
        sample_model=sample_model,
        kernels={
            'Hamming': HammingKernel(),
        },
        stein_operators={
            r'$\mathrm{ZS}_1$': OldStyleZanellaSteinOperator(target_model, ['insert', 'replace', 'pop'], 1),
            r'$\mathrm{ZS}_\infty$': OldStyleZanellaSteinOperator(target_model, ['insert', 'replace', 'pop'], None),
            r"$\mathrm{ZS}_\infty'$": OldStyleZanellaSteinOperator(target_model, ['replace'], None),
            'Gibbs': GibbsSteinOperator(target_model),
        },
        N_p=N_q,
        N_q=N_q,
        N_bootstrap_D=100,
        N_bootstrap_MMD=100,
        N_bootstrap_wild=1_000,
        do_wild_bootstrap=False,
        verify_wild_bootstrap=False,
        # fixed_length=E_length,
    )


def main():
    rng = np.random.default_rng(3876632683 + 1_100)
    experiment = Experiment(
        constructor=lambda params: get_point(rng=rng, **params),
        baseline_parameters={
            'alpha_target': 1,
            'alpha_sample': 2,
            'N_q': 50,
            'n_states': 10,
            'E_length': 8,
        },
        parameters_to_vary={
            # 'alpha_sample': np.linspace(0, 1, 11),
            # 'alpha_target': np.linspace(0, 1, 11),
            # 'N_q': np.logspace(1, 2, 5, dtype=int),
            'n_states': np.geomspace(2, 20, 5, dtype=int),
            # 'E_length': np.linspace(2, 80, 10, dtype=int),
        },
    )

    with timer():
        non_binary_iid_output = experiment.run(rng, client=None)
    experiment_name = 'exp_non_binary_iid'
    plot_output(
        non_binary_iid_output,
        experiment_name=experiment_name,
        log_scale={'N_q', 'n_states'},
    )
    for variable, results in non_binary_iid_output.items():
        results.test_error_rates.to_csv(f'../figures/{experiment_name}_error_rates_by_{variable}.csv')


if __name__ == '__main__':
    main()
