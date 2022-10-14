# fix relative imports
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    __package__ = Path(__file__).parent.name
# end of relative imports fix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .deprecated_experiments import Experiment
from .deprecated_experiments import ExperimentPoint
from .kernels import HammingKernel
from .models import NgramModel
from .stein_operators import OldStyleZanellaSteinOperator
from .utils import timer


def get_point(*, rng, alpha_target, alpha_sample, E_length, N_q, n_states, neigh_sizes):
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
            **{
                r'$\mathrm{ZS}_{' + str(i) + ',dt}$': OldStyleZanellaSteinOperator(target_model,
                                                                                   ['insert', 'replace', 'pop', 'drop_tail'], i)
                for i in neigh_sizes
            },
            **{
                r'$\mathrm{ZS}_{' + str(i) + '}$': OldStyleZanellaSteinOperator(target_model, ['insert', 'replace', 'pop'], i)
                for i in neigh_sizes
            },
            **{
                r"$\mathrm{ZS}_{" + str(i) + "}'$": OldStyleZanellaSteinOperator(target_model, ['replace'], i)
                for i in neigh_sizes
            },
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
    neigh_sizes = np.linspace(1, 10, 5, dtype=int)
    rng = np.random.default_rng(3876632683 + 1_100)
    experiment = Experiment(
        constructor=lambda params: get_point(rng=rng, **params),
        baseline_parameters={
            'alpha_target': 1,
            'alpha_sample': 1.5,
            'N_q': 50,
            'n_states': 10,
            'E_length': 8,
            'neigh_sizes': neigh_sizes,
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
        neighbourhood_size_output = experiment.run(rng, client=None)

    experiment_name = 'exp_neighbourhood_size'
    operators = {
        # **{
        #     r'$\mathrm{ZS}_{' + str(i) + '}$': (i, r'$\mathrm{ZS}_i$')
        #     for i in range(1, 5+1)
        # },
        **{
            r"$\mathrm{ZS}_{" + str(i) + "}'$": (i, r"$\mathrm{ZS}_i'$")
            for i in neigh_sizes
        },
    }
    for variable, results in neighbourhood_size_output.items():
        data = pd.DataFrame(index=neigh_sizes)
        data.index.name = 'i'
        for value in results.test_error_rates.index:
            # mmd_original_chart_label = f'MMD type II error (k=Hamming)'
            # mmd_full_chart_label = f'MMD type II error (k=Hamming, {variable}={value})'
            # data[mmd_full_chart_label] = results.test_error_rates[mmd_original_chart_label].loc[value]
            for op_name, (op_param, op_chart_label) in operators.items():
                original_chart_label = f'Stein type II error (k=Hamming, op={op_name})'
                full_chart_label = f'Stein type II error (k=Hamming, op={op_chart_label}, {variable}={value})'
                if full_chart_label not in data.columns:
                    data[full_chart_label] = np.nan
                data[full_chart_label].loc[op_param] = results.test_error_rates[original_chart_label].loc[value]
        sns.lineplot(data=data)
        plt.gca().set_ylim([-.02, 1.02])
        plt.title(f'Experiment: {experiment_name}')
        plt.gca().axhline(0.05, label=r'$\alpha$', color='gray', ls='--')
        plt.gca().axhline(0.95, label=r'$1-\alpha$', color='gray', ls='--')
        if experiment_name is not None:
            plt.savefig(f'../figures/{experiment_name}_error_rates_by_neigh_size.pdf')
        plt.show()
    # plot_output(
    #     neighbourhood_size_output,
    #     experiment_name=experiment_name,
    #     log_scale={'N_q', 'n_states'},
    # )
    for variable, results in neighbourhood_size_output.items():
        results.test_error_rates.to_csv(f'../figures/{experiment_name}_error_rates_by_{variable}.csv')


if __name__ == '__main__':
    main()
