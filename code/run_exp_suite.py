# fix relative imports
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    __package__ = Path(__file__).parent.name
# end of relative imports fix

import datetime

import numpy as np
import pandas as pd

from .kernels import HammingKernel
from .kernels import GaussianKernel
from .kernels import ContiguousSubsequenceKernel
from .kernels import PrefixKernel
from .kernels import FixedLengthKernel
from .kernels import SteinKernel
from .models import NgramModel
from .models import RandomNgramModel
from .models import RandomNgramModelWithFixedInitDist
from .models import PoissonBernoulliModel
from .models import RandomWalk
from .models import RandomWalkWithHolding
from .models import RandomWalkWithMemory
from .models import RandomUniformIID
from .models import CTMCAdjustedModel
from .models import BetaBinomialModel
from .models import SimpleMRFModel
from .utils import get_empirical_model, slugify_unsafe, slugify
from .utils import timer
from .utils import Seed, extend_seed, get_rng
from .stein_operators import ZanellaSteinOperator
from .stein_operators import MaxLikelihoodZSOperator
from .stein_operators import SumOfSteinOperators
from .stein_operators import GibbsSteinOperator
from .hypothesis_tests import SteinTestWildBootstrap
from .hypothesis_tests import SteinTestParametricBootstrap
from .hypothesis_tests import MMDTestParametricBootstrap
from .hypothesis_tests import AggregatedTestParametricBootstrap
from .hypothesis_tests import LikelihoodRatioTest
from .experiments import run_test
from .experiments import HeldTest
from .actions import SubActionSet
from .actions import PreActionSet
from .actions import MaxProbSubActionSet
from .actions import MinProbSubActionSet
from .actions import InsertActionSet
from .actions import DelActionSet
from .actions import IncrDecrActionSet
from .actions import UnionOfActionSets


def get_tests(*, seed: Seed, target_model, sample_model, N_bootstrap, N_sequences, desired_level, alphabet_is_ordered,
              markov_order,
              prefer_wild_bootstrap: bool):
    n_states = len(target_model.alphabet)
    test_kwargs = dict(n_bootstrap=N_bootstrap, desired_level=desired_level, sample_size=N_sequences)
    hold = lambda con: lambda *args, **kwargs: HeldTest(lambda rng: con(*args, rng=rng, **kwargs))

    if prefer_wild_bootstrap:
        mmd_test_for_degenerate_case = MMDTestParametricBootstrap
        stein_test_for_degenerate_case = SteinTestWildBootstrap
    else:
        mmd_test_for_degenerate_case = MMDTestParametricBootstrap
        stein_test_for_degenerate_case = SteinTestParametricBootstrap

    mmd_test_supports_wild_bootstrap = False
    stein_test_supports_wild_bootstrap = True

    stein_ops = {
        'Gibbs': dict(
            version='1',
            op=GibbsSteinOperator(model=target_model),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_{\infty,mpf}$": dict(
            version='1',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                        | InsertActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=None)
                ),
                kappa='mpf',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_{\infty,mh}$": dict(
            version='1',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                        | InsertActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=None)
                ),
                kappa='mh',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_{\infty,b}$": dict(
            version='1',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                        | InsertActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=None)
                ),
                kappa='barker',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_{\infty,b}'$": dict(
            version='1',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet),
                kappa='barker',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_1$": dict(
            version='3',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        SubActionSet(complement=True, min_ix=0, max_ix=1, xs=target_model.alphabet)
                        | InsertActionSet(complement=True, min_ix=0, max_ix=1, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=1)
                ),
                kappa='barker',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_1'$": dict(
            version='2',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=SubActionSet(complement=True, min_ix=0, max_ix=1, xs=target_model.alphabet),
                kappa='barker',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_1''$": dict(
            version='2',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        InsertActionSet(complement=True, min_ix=0, max_ix=1, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=1)
                ),
                kappa='barker',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_3$": dict(
            version='3',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        SubActionSet(complement=True, min_ix=0, max_ix=3, xs=target_model.alphabet)
                        | InsertActionSet(complement=True, min_ix=0, max_ix=3, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=3)
                ),
                kappa='barker',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_3'$": dict(
            version='2',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=SubActionSet(complement=True, min_ix=0, max_ix=3, xs=target_model.alphabet),
                kappa='barker',
            ),
            is_stein=True,
        ),
        r"$\mathrm{ZS}_3''$": dict(
            version='2',
            op=ZanellaSteinOperator(
                model=target_model,
                action_set=(
                        InsertActionSet(complement=True, min_ix=0, max_ix=3, xs=target_model.alphabet)
                        | DelActionSet(complement=True, min_ix=0, max_ix=3)
                ),
                kappa='barker',
            ),
            is_stein=True,
        ),
    }

    for n in [5, 7, 9]:
        stein_ops |= {
            r"$\mathrm{ZS}_" + f"{n}$": dict(
                version='3',
                op=ZanellaSteinOperator(
                    model=target_model,
                    action_set=(
                            SubActionSet(complement=True, min_ix=0, max_ix=n, xs=target_model.alphabet)
                            | InsertActionSet(complement=True, min_ix=0, max_ix=n, xs=target_model.alphabet)
                            | DelActionSet(complement=True, min_ix=0, max_ix=n)
                    ),
                    kappa='barker',
                ),
                is_stein=True,
            ),
            r"$\mathrm{ZS}_" + f"{n}'$": dict(
                version='2',
                op=ZanellaSteinOperator(
                    model=target_model,
                    action_set=SubActionSet(complement=True, min_ix=0, max_ix=n, xs=target_model.alphabet),
                    kappa='barker',
                ),
                is_stein=True,
            ),
            r"$\mathrm{ZS}_" + f"{n}''$": dict(
                version='2',
                op=ZanellaSteinOperator(
                    model=target_model,
                    action_set=(
                            InsertActionSet(complement=True, min_ix=0, max_ix=n, xs=target_model.alphabet)
                            | DelActionSet(complement=True, min_ix=0, max_ix=n)
                    ),
                    kappa='barker',
                ),
                is_stein=True,
            ),
        }

    for n in [1, 2, 3]:
        stein_ops |= {
            r"$\mathrm{ZS}_1'$ with incr decr (" + f"{n})": dict(
                version='2',
                op=ZanellaSteinOperator(
                    model=target_model,
                    action_set=IncrDecrActionSet(
                        complement=True,
                        min_ix=0, max_ix=1,
                        the_range=n,
                        xs=target_model.alphabet,
                    ),
                    kappa='barker',
                ),
                is_stein=True,
            ),
            r"$\mathrm{ZS}_3'$ with incr decr (" + f"{n})": dict(
                version='2',
                op=ZanellaSteinOperator(
                    model=target_model,
                    action_set=IncrDecrActionSet(
                        complement=True,
                        min_ix=0, max_ix=3,
                        the_range=n,
                        xs=target_model.alphabet,
                    ),
                    kappa='barker',
                ),
                is_stein=True,
            ),
        }

    stein_ops |= {
        # r"$\mathrm{ZS}_{3,mpf}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet),
        #         kappa='mpf',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{3,mh}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet),
        #         kappa='mh',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{7,mpf}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=SubActionSet(complement=True, min_ix=0, max_ix=6, xs=target_model.alphabet),
        #         kappa='mpf',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{7,mh}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=SubActionSet(complement=True, min_ix=0, max_ix=6, xs=target_model.alphabet),
        #         kappa='mh',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{\infty,mpf}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet),
        #         kappa='mpf',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{10,mpf}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=SubActionSet(complement=True, min_ix=0, max_ix=9, xs=target_model.alphabet),
        #         kappa='mpf',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{15,geom,mpf}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=UnionOfActionSets(*[
        #             SubActionSet(complement=True, min_ix=i, max_ix=i, xs=target_model.alphabet)
        #             for i in np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 31]) - 1
        #         ]),
        #         kappa='mpf',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{\infty,mpf,mp5}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
        #                 | MaxProbSubActionSet(model=target_model, tail=True, min_num_to_sub=5, max_num_to_sub=5)
        #         ),
        #         kappa='mpf',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{\infty,mpf,mp5}$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
        #                 | InsertActionSet(complement=True, min_ix=0, max_ix=None, xs=target_model.alphabet)
        #                 | DelActionSet(complement=True, min_ix=0, max_ix=None)
        #                 | MaxProbSubActionSet(model=target_model, tail=True, min_num_to_sub=5, max_num_to_sub=5)
        #         ),
        #         kappa='mpf',
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{3,mp5}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
        #                 | MaxProbSubActionSet(model=target_model, tail=True, min_num_to_sub=5, max_num_to_sub=5)
        #         ),
        #         kappa='barker',
        #     ),
        #     is_stein=False,
        # ),
        # r"$\mathrm{ZS}_{3,minp5}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
        #                 | MinProbSubActionSet(model=target_model, tail=True, min_num_to_sub=5, max_num_to_sub=5)
        #         ),
        #         kappa='barker',
        #     ),
        #     is_stein=False,
        # ),
        # r"$\mathrm{ZS}_{3,mp5}''$": dict(
        #     version='1',
        #     op=SumOfSteinOperators(
        #         ZanellaSteinOperator(
        #             model=target_model,
        #             action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
        #             ),
        #             kappa='barker',
        #         ),
        #         MaxLikelihoodZSOperator(
        #             model=target_model,
        #             num_to_sub=5,
        #             kappa='barker',
        #         )
        #     ),
        #     is_stein=True,
        # ),
        # r"$\mathrm{ZS}_{1,pre8}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=0, xs=target_model.alphabet)
        #                 | PreActionSet(complement=False, min_ix=1, max_ix=8)
        #         ),
        #         kappa='barker',
        #     ),
        #     is_stein=False,
        # ),
        # r"$\mathrm{ZS}_{3,pre8}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
        #                 | PreActionSet(complement=False, min_ix=1, max_ix=8)
        #         ),
        #         kappa='barker',
        #     ),
        #     is_stein=False,
        # ),
        # r"$\mathrm{ZS}_{1,pre4}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=0, xs=target_model.alphabet)
        #                 | PreActionSet(complement=False, min_ix=1, max_ix=4)
        #         ),
        #         kappa='barker',
        #     ),
        #     is_stein=False,
        # ),
        # r"$\mathrm{ZS}_{3,pre4}'$": dict(
        #     version='1',
        #     op=ZanellaSteinOperator(
        #         model=target_model,
        #         action_set=(
        #                 SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
        #                 | PreActionSet(complement=False, min_ix=1, max_ix=4)
        #         ),
        #         kappa='barker',
        #     ),
        #     is_stein=False,
        # ),
    }

    tests = dict()

    def make_agg_mmd_test(rng, kernel, desired_level, **kwargs):
        kwargs['n_bootstrap'] *= 4
        N = 30
        unif_weight = -np.log(1 / N)
        tests_and_weights = [
            (MMDTestParametricBootstrap(
                rng=rng,
                kernel=FixedLengthKernel(length=n, underlying_kernel=kernel),
                desired_level=desired_level,
                **kwargs
            ), unif_weight)
            for n in range(1, N + 1)
        ]
        return AggregatedTestParametricBootstrap(
            tests_and_weights=tests_and_weights,
            desired_level=desired_level,
            rng=rng,
            target_model=target_model,
        )

    def make_agg_stein_test(rng, kernel, desired_level, target_model, **kwargs):
        kwargs['n_bootstrap'] *= 4
        action_sets_and_weights = [
            (
                SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
                , 1
            ),
            (
                SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
                | MaxProbSubActionSet(model=target_model, tail=True, min_num_to_sub=5, max_num_to_sub=5)
                , 1
            ),
            (
                SubActionSet(complement=True, min_ix=0, max_ix=2, xs=target_model.alphabet)
                | PreActionSet(complement=False, min_ix=1, max_ix=8)
                , 1
            ),
        ]
        weight_adj = np.log(sum(np.exp(-w) for _, w in action_sets_and_weights))
        action_sets_and_weights = [(s, w + weight_adj) for s, w in action_sets_and_weights]

        tests_and_weights = [
            (SteinTestParametricBootstrap(
                kernel=kernel,
                stein_op=ZanellaSteinOperator(
                    model=target_model,
                    action_set=action_set,
                    kappa='barker',
                ),
                target_model=target_model,
                rng=rng,
                desired_level=desired_level,
                **kwargs
            ),
             weight)
            for action_set, weight in action_sets_and_weights
        ]
        return AggregatedTestParametricBootstrap(
            tests_and_weights=tests_and_weights,
            desired_level=desired_level,
            rng=rng,
            target_model=target_model,
        )

    if alphabet_is_ordered:
        basic_kernel = 'ExpDist', GaussianKernel(n_states=n_states)
        better_kernel = 'CSK', ContiguousSubsequenceKernel(min_length=markov_order + 1, max_length=markov_order + 1)
    else:
        basic_kernel = 'ExpDist', HammingKernel()
        better_kernel = 'CSK', ContiguousSubsequenceKernel(min_length=markov_order + 1, max_length=markov_order + 1)

    for kernel in [basic_kernel, better_kernel]:
        tests |= {
            # f'Stein(k={kernel[0]}, op={k}, $\\alpha\\gets\\alpha/2$)': dict(
            #     version=f"1.{v['version']}",
            #     test=hold(stein_test_for_degenerate_case if v['is_stein'] else SteinTestParametricBootstrap)(
            #         kernel=kernel[1],
            #         stein_op=v['op'],
            #         target_model=target_model,
            #         **(test_kwargs | {'desired_level': desired_level / 2})
            #     ),
            #     supports_wild_bootstrap=stein_test_supports_wild_bootstrap,
            # )
            # for k, v in {
            #     r"$\mathrm{ZS}_5'$": dict(
            #         version='3',
            #         op=ZanellaSteinOperator(
            #             model=target_model,
            #             action_set=SubActionSet(complement=True, min_ix=0, max_ix=5, xs=target_model.alphabet),
            #             kappa='barker',
            #         ),
            #         is_stein=True,
            #     ),
            #     r"$\mathrm{ZS}_9'$": dict(
            #         version='2',
            #         op=ZanellaSteinOperator(
            #             model=target_model,
            #             action_set=SubActionSet(complement=True, min_ix=0, max_ix=9, xs=target_model.alphabet),
            #             kappa='barker',
            #         ),
            #         is_stein=True,
            #     ),
            # }.items()
        }

    kernels = [(basic_kernel, False), (better_kernel, True)]
    del basic_kernel
    del better_kernel
    for kernel, is_expensive in kernels:
        for n_mmd_samples in [10, 33, 100, 333, 1000]:
            tests |= {
                f'MMD(k={kernel[0]}, n={n_mmd_samples})': dict(
                    version='3',
                    test=hold(mmd_test_for_degenerate_case)(
                        kernel=kernel[1],
                        n_mmd_samples=n_mmd_samples,
                        target_model=target_model,
                        **test_kwargs
                    ),
                    supports_wild_bootstrap=mmd_test_supports_wild_bootstrap,
                ),
            }

        tests |= {
            f'Stein(k={kernel[0]}, op={k})': dict(
                version=f"1.{v['version']}",
                test=hold(stein_test_for_degenerate_case if v['is_stein'] else SteinTestParametricBootstrap)(
                    kernel=kernel[1],
                    stein_op=v['op'],
                    target_model=target_model,
                    **test_kwargs
                ),
                supports_wild_bootstrap=stein_test_supports_wild_bootstrap,
            )
            for k, v in stein_ops.items()
        }

        # tests |= {
        #     f'Stein-Wild(k={kernel[0]}, op={k})': dict(
        #         version=f"1.{v['version']}",
        #         test=hold(SteinTestWildBootstrap)(
        #             kernel=kernel[1],
        #             stein_op=v['op'],
        #             target_model=target_model,
        #             **test_kwargs
        #         ),
        #         supports_wild_bootstrap=False,  # do not make cache file depend on wildness
        #     )
        #     for k, v in stein_ops.items()
        # }

        # for n_mmd_samples in [100]:
        #     tests |= {
        #         f'Agg-MMD(k={kernel[0]}, n={n_mmd_samples})': dict(
        #             version='3',
        #             test=hold(make_agg_mmd_test)(
        #                 kernel=kernel[1],
        #                 n_mmd_samples=n_mmd_samples,
        #                 target_model=target_model,
        #                 **test_kwargs
        #             ),
        #             supports_wild_bootstrap=False,
        #         ),
        #     }

        tests |= {
            # f"""Stein-MMD(k={kernel[0]}, op=$\\{r"mathrm{ZS}_{3,mp5}'$"}, n=100)""": dict(
            #     version='1',
            #     test=hold(mmd_test_for_degenerate_case)(
            #         kernel=SteinKernel(
            #             underlying_kernel=kernel[1],
            #             stein_operator=ZanellaSteinOperator(
            #                 model=target_model,
            #                 action_set=(
            #                         SubActionSet(complement=True, min_ix=0, max_ix=2,
            #                                      xs=target_model.alphabet)
            #                         | MaxProbSubActionSet(model=target_model, tail=True,
            #                                               min_num_to_sub=5,
            #                                               max_num_to_sub=5)
            #                 ),
            #                 kappa='barker',
            #             ),
            #         ),
            #         n_mmd_samples=100,
            #         target_model=target_model,
            #         **test_kwargs
            #     ),
            #     supports_wild_bootstrap=False,
            # ),
        }

        # mmd_ap_Ts = [1, 3, 5, 10]
        mmd_ap_Ts = [10]
        for mmd_ap_T in mmd_ap_Ts:
            tests |= {
                # f"""MMD-Ap(k={kernel[0]}, op=$\\{r"mathrm{ZS}_{3,mp5}'$"}, n=100, T={mmd_ap_T})""": dict(
                #     version='2',
                #     test=hold(MMDTestParametricBootstrap)(
                #         kernel=kernel[1],
                #         n_mmd_samples=100,
                #         target_model=CTMCAdjustedModel(
                #             underlying_model=target_model,
                #             action_set=(
                #                     SubActionSet(complement=True, min_ix=0, max_ix=2,
                #                                  xs=target_model.alphabet)
                #                     | MaxProbSubActionSet(model=target_model, tail=True,
                #                                           min_num_to_sub=5,
                #                                           max_num_to_sub=5)
                #             ),
                #             kappa='barker',
                #             T=mmd_ap_T,
                #         ),
                #         **test_kwargs
                #     ),
                #     supports_wild_bootstrap=False,
                # ),
                # f"""MMD-Ap(k={kernel[0]}, op=$\\{r"mathrm{ZS}_3'$"}, n=100, T={mmd_ap_T})""": dict(
                #     version='2',
                #     test=hold(MMDTestParametricBootstrap)(
                #         kernel=kernel[1],
                #         n_mmd_samples=100,
                #         target_model=CTMCAdjustedModel(
                #             underlying_model=target_model,
                #             action_set=(
                #                 SubActionSet(complement=True, min_ix=0, max_ix=2,
                #                              xs=target_model.alphabet)
                #             ),
                #             kappa='barker',
                #             T=mmd_ap_T,
                #         ),
                #         **test_kwargs
                #     ),
                #     supports_wild_bootstrap=False,
                # ),
            }

        tests |= {
            f'MMD(k=Pre({kernel[0]},5), n=100)': dict(
                version='1',
                test=hold(mmd_test_for_degenerate_case)(
                    kernel=PrefixKernel(prefix_size=5, underlying_kernel=kernel[1]),
                    n_mmd_samples=100,
                    target_model=target_model,
                    **test_kwargs
                ),
                supports_wild_bootstrap=mmd_test_supports_wild_bootstrap,
            ),
        }

        tests |= {
            f'Agg-Stein(k={kernel[0]})': dict(
                version='5',
                test=hold(make_agg_stein_test)(
                    kernel=kernel[1],
                    target_model=target_model,
                    **test_kwargs
                ),
                supports_wild_bootstrap=False,
            ),
        }

    tests |= {
        r'LR($k=k_\mathrm{true}$)': dict(
            version='1',
            test=hold(LikelihoodRatioTest)(
                target_model=target_model,
                get_mle_model=lambda xs: NgramModel.fit(n_in_ngram=markov_order, n_states=n_states, sequences=xs),
                **test_kwargs,
            ),
            supports_wild_bootstrap=False,
        ),
        r'LR($k=k_\mathrm{true}+1$)': dict(
            version='1',
            test=hold(LikelihoodRatioTest)(
                target_model=target_model,
                get_mle_model=lambda xs: NgramModel.fit(n_in_ngram=markov_order + 1, n_states=n_states, sequences=xs),
                **test_kwargs,
            ),
            supports_wild_bootstrap=False,
        ),
        r'LR($k=\infty$)': dict(
            version='1',
            test=hold(LikelihoodRatioTest)(
                target_model=target_model,
                get_mle_model=get_empirical_model,
                **test_kwargs,
            ),
            supports_wild_bootstrap=False,
        ),
        r'LR(oracle)': dict(
            version='1',
            test=hold(LikelihoodRatioTest)(
                target_model=target_model,
                get_mle_model=lambda xs: sample_model,
                **test_kwargs,
            ),
            supports_wild_bootstrap=False,
        ),
    }

    return tests


def get_scenarios(*, seed: Seed):
    scenarios = {
        'Binary sequences: Few long i.i.d. sequences': dict(
            version='2',
            target_model=PoissonBernoulliModel(20, 0.6),
            sample_model=PoissonBernoulliModel(20, 0.4),
            N_sequences=10,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=0,
        ),
        'Binary sequences: True distribution is not i.i.d. sequence': dict(
            version='2',
            target_model=RandomUniformIID(n_states=2, E_length=20),
            sample_model=RandomWalk(n_states=2, E_length=20),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=0,
        ),
        'Random walk: Many short sequences': dict(
            version='3',
            target_model=RandomWalk(n_states=8, E_length=8),
            sample_model=RandomWalkWithHolding(
                n_states=8, E_length=8,
                holding_probability=0.2, n_holding_states=8,
            ),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=True,
            markov_order=1,
        ),
        'Random walk: Few long sequences': dict(
            version='2',
            target_model=RandomWalk(n_states=8, E_length=30),
            sample_model=RandomWalkWithHolding(
                n_states=8, E_length=30,
                holding_probability=0.2, n_holding_states=8,
            ),
            N_sequences=10,
            desired_level=0.05,
            alphabet_is_ordered=True,
            markov_order=1,
        ),
        'Random walk with memory: Many short sequences': dict(
            version='1',
            target_model=RandomWalkWithMemory(n_states=10, E_length=8, memory=0.95),
            sample_model=RandomWalkWithMemory(n_states=10, E_length=8, memory=0.05),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=True,
            markov_order=2,
        ),
        'Random walk with memory: Few long sequences': dict(
            version='1',
            target_model=RandomWalkWithMemory(n_states=10, E_length=30, memory=0.95),
            sample_model=RandomWalkWithMemory(n_states=10, E_length=30, memory=0.05),
            N_sequences=8,
            desired_level=0.05,
            alphabet_is_ordered=True,
            markov_order=2,
        ),
        'Random 2nd-order MC: Many short sequences': dict(
            version='2',
            target_model=RandomNgramModel(
                rng=get_rng(seed, 'random-ngram', 1),
                n_in_ngram=2,
                n_states=10,
                E_length=8,
                concentration_param=1,
            ),
            sample_model=RandomNgramModel(
                rng=get_rng(seed, 'random-ngram', 2),
                n_in_ngram=2,
                n_states=10,
                E_length=8,
                concentration_param=1,
            ),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=2,
        ),
        'Random 2nd-order MC: Few long sequences': dict(
            version='2',
            target_model=RandomNgramModel(
                rng=get_rng(seed, 'random-ngram', 1, nth_usage=2),
                n_in_ngram=2,
                n_states=10,
                E_length=20,
                concentration_param=1,
            ),
            sample_model=RandomNgramModel(
                rng=get_rng(seed, 'random-ngram', 2, nth_usage=2),
                n_in_ngram=2,
                n_states=10,
                E_length=20,
                concentration_param=1,
            ),
            N_sequences=8,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=2,
        ),
        'Random 2nd-order MC: Few short sequences': dict(
            version='3',
            target_model=RandomNgramModel(
                rng=get_rng(seed, 'random-ngram', 1, nth_usage=3),
                n_in_ngram=2,
                n_states=10,
                E_length=8,
                concentration_param=1,
            ),
            sample_model=RandomNgramModel(
                rng=get_rng(seed, 'random-ngram', 2, nth_usage=3),
                n_in_ngram=2,
                n_states=10,
                E_length=8,
                concentration_param=1,
            ),
            N_sequences=8,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=2,
        ),
        'Random MC w/ varied initial dist: Many short sequences': dict(
            version='5',
            target_model=RandomNgramModelWithFixedInitDist(
                rng=get_rng(seed, 'random-ngram', 3),
                n_in_ngram=1,
                n_states=10,
                E_length=8,
                concentration_param=1,
                initial_dist=np.ones(10) / 10,
            ),
            sample_model=RandomNgramModelWithFixedInitDist(
                rng=get_rng(seed, 'random-ngram', 3, nth_usage=2),
                n_in_ngram=1,
                n_states=10,
                E_length=8,
                concentration_param=1,
                initial_dist=(np.ones(10) / 10 + np.block([np.ones(2), np.zeros(8)]) / 2) / 2,
            ),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=1,
        ),
        'Random MC w/ varied initial dist: Few long sequences': dict(
            version='5',
            target_model=RandomNgramModelWithFixedInitDist(
                rng=get_rng(seed, 'random-ngram', 3, nth_usage=3),
                n_in_ngram=1,
                n_states=10,
                E_length=20,
                concentration_param=1,
                initial_dist=np.ones(10) / 10,
            ),
            sample_model=RandomNgramModelWithFixedInitDist(
                rng=get_rng(seed, 'random-ngram', 3, nth_usage=4),
                n_in_ngram=1,
                n_states=10,
                E_length=20,
                concentration_param=1,
                initial_dist=(np.ones(10) / 10 + np.block([np.ones(2), np.zeros(8)]) / 2) / 2,
            ),
            N_sequences=8,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=1,
        ),
        'Random MC w/ varied length dist': dict(
            version='3',
            target_model=RandomNgramModelWithFixedInitDist(
                rng=get_rng(seed, 'random-ngram', 3, nth_usage=5),
                n_in_ngram=1,
                n_states=10,
                E_length=8,
                concentration_param=1,
                initial_dist=np.ones(10) / 10,
            ),
            sample_model=RandomNgramModelWithFixedInitDist(
                rng=get_rng(seed, 'random-ngram', 3, nth_usage=6),
                n_in_ngram=1,
                n_states=10,
                E_length=20,
                concentration_param=1,
                initial_dist=np.ones(10) / 10,
            ),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=2,
        ),
        'Binary sequences, sample is non-Markov': dict(
            version='1',
            target_model=PoissonBernoulliModel(30, 0.5),
            sample_model=BetaBinomialModel(
                E_length=30,
                a=5,
                b=5,
            ),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=2,
        ),
        'Simple MRF': dict(
            version='1',
            target_model=SimpleMRFModel(
                n_states=5,
                concentration_param=0,
                max_length=40,
                length_potential=0,
            ),
            sample_model=SimpleMRFModel(
                n_states=5,
                concentration_param=1,
                max_length=40,
                length_potential=0,
            ),
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=2,
        ),
    }
    model_for_level_check = RandomNgramModel(
        rng=get_rng(seed, 'random-ngram', 1, nth_usage=4),
        n_in_ngram=2,
        n_states=10,
        E_length=8,
        concentration_param=1,
    )
    scenarios |= {
        'Random 2nd-order MC: Level check': dict(
            version='2',
            target_model=model_for_level_check,
            sample_model=model_for_level_check,
            N_sequences=30,
            desired_level=0.05,
            alphabet_is_ordered=False,
            markov_order=2,
        ),
    }
    return scenarios


def run_experiment(*, test_run):
    suite_version = '3'  # there is unlikely to be a reason to bump this beyond v3
    if test_run:
        noise_parameters = {
            'N_indep_tests': 1,
            'N_repeats_per_test': 1,
            'N_bootstrap': 10,
            'prefer_wild_bootstrap': True,
        }
    else:
        noise_parameters = {
            'N_indep_tests': 4,
            'N_repeats_per_test': 100,
            # 'N_indep_tests': 1,
            # 'N_repeats_per_test': 50,
            'N_bootstrap': 100,
            'prefer_wild_bootstrap': False,
        }

    root_seed = 9824740821212930092259595777667619796, ''
    results = dict()

    def go():
        scenarios = get_scenarios(seed=extend_seed(root_seed, 'scenarios'))
        if test_run:
            scenarios = {k: scenarios[k] for k in list(scenarios.keys())[:2]}
        for scenario_name, scenario in scenarios.items():
            if not test_run:
                print(scenario_name)
            results[scenario_name] = dict()
            sample_model = scenario['sample_model']
            N_sequences = scenario['N_sequences']
            tests = get_tests(
                seed=extend_seed(root_seed, f'{scenario_name}/tests'),
                N_sequences=N_sequences,
                N_bootstrap=noise_parameters['N_bootstrap'],
                desired_level=scenario['desired_level'] if not test_run else 0.5,
                alphabet_is_ordered=scenario['alphabet_is_ordered'],
                target_model=scenario['target_model'],
                sample_model=sample_model,
                markov_order=scenario['markov_order'],
                prefer_wild_bootstrap=noise_parameters['prefer_wild_bootstrap'],
            )
            if test_run:
                tests = {k: tests[k] for k in list(tests.keys())[:2]}
            for test_name, test in tests.items():
                if not test_run:
                    print('  ' + test_name)
                suite_part = f"suite-{suite_version}"
                scenario_part = f"{slugify(scenario_name)}-{scenario['version']}"
                test_part = f"{slugify(test_name)}-{test['version']}"
                if test['supports_wild_bootstrap']:
                    noise_param_part = f"{noise_parameters['N_bootstrap']}_{noise_parameters['prefer_wild_bootstrap']}"
                else:
                    noise_param_part = f"{noise_parameters['N_bootstrap']}_x"
                cache_path = f'../figures/suite/{suite_part}__{scenario_part}__{test_part}__{noise_param_part}.pickle'  # to pass to run_test
                run_test_ = lambda: run_test(
                    seed=extend_seed(root_seed, f'{scenario_name}/{test_name}'),
                    test=test['test'],
                    sample_model=sample_model,
                    N_sequences=N_sequences,
                    N_indep_tests=noise_parameters['N_indep_tests'],
                    N_repeats_per_test=noise_parameters['N_repeats_per_test'],
                    test_run=test_run,
                    cache_path=cache_path,
                )
                if test_run:
                    this_result = run_test_()
                else:
                    this_result = run_test_()
                if not test_run:
                    print('   ' + str(this_result['rejection_rate']))
                results[scenario_name][test_name] = this_result

    if test_run:
        go()
    else:
        with timer():
            go()
    return results


scenarios_to_average_over = [
    "Binary sequences: Few long i.i.d. sequences",
    "Binary sequences: True distribution is not i.i.d. sequence",
    "Random walk: Many short sequences",
    "Random walk: Few long sequences",
    "Random walk with memory: Many short sequences",
    "Random walk with memory: Few long sequences",
    "Random 2nd-order MC: Many short sequences",
    "Random 2nd-order MC: Few long sequences",
    "Random 2nd-order MC: Few short sequences",
    "Random MC w/ varied initial dist: Many short sequences",
    "Random MC w/ varied initial dist: Few long sequences",
    "Random MC w/ varied length dist",
]


def main():
    results = run_experiment(test_run=False)
    measured_total_suite_seconds = sum(sum(v2['total_seconds'] for v2 in v.values()) for v in results.values())
    print(
        f"approx time to run entire suite from scratch: {datetime.timedelta(seconds=measured_total_suite_seconds)}")
    measured_construction_seconds = sum(
        sum(v2['avg_test_construction_seconds'] for v2 in v.values()) for v in results.values())
    print(f"approx time to construct all tests once: {datetime.timedelta(seconds=measured_construction_seconds)}")
    variables_to_write = [
        'rejection_rate',
        'avg_total_seconds',
    ]
    seriess_to_write = {}
    for varname in variables_to_write:
        to_write = {k: {k2: v2[varname] for k2, v2 in v.items()} for k, v in results.items()}
        df = pd.DataFrame(to_write)
        df.index.name = 'Test'
        safeify = lambda x: x.replace(',', ':').replace('$', '_').replace('\\', '_')
        df.index = df.index.map(lambda x: '{' + safeify(x) + '}')
        df.columns.name = 'Scenario'
        df["suite_mean"] = df[scenarios_to_average_over].T.mean()
        df["suite_median"] = df[scenarios_to_average_over].T.median()
        df.columns = df.columns.map(lambda x: x.replace(',', ':'))
        filename = f'../figures/suite--{varname}.csv'
        df.to_csv(filename)
        filename = f'../figures/suite--{varname}.T.csv'
        df.T.to_csv(filename)
        series = df.unstack()
        series.name = varname
        seriess_to_write[varname] = series
    df_to_write = pd.DataFrame(seriess_to_write)
    filename = f'../figures/suite-output.csv'
    df_to_write.to_csv(filename)


if __name__ == '__main__':
    main()
