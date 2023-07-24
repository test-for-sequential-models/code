# __file__ = '/Users/jeromebaum/repos/ucl-2021-msc-thesis/thesis_code/run_exp_language.py'

from pathlib import Path
import sys

target_path = Path(__file__).parent.parent
if str(target_path) not in sys.path:
    sys.path.insert(0, str(target_path))

import os
import shutil
import tempfile
from pathlib import Path
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
from transformers import set_seed
import torch
from torch import nn
import tqdm.auto as tqdm
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from thesis_code.actions import ActionSet
from thesis_code.sequences import Sequence
from thesis_code.models import SequenceDist
from thesis_code.models import apply_basic_edit
from thesis_code.models import BasicEditType
from thesis_code.stein_operators import ZanellaSteinOperator
from thesis_code.hypothesis_tests import MMDTestParametricBootstrap
from thesis_code.hypothesis_tests import SteinTestParametricBootstrap
from thesis_code.hypothesis_tests import MMDTestWildBootstrap
from thesis_code.hypothesis_tests import SteinTestWildBootstrap
from thesis_code.kernels import ContiguousSubsequenceKernel
from thesis_code.experiments import run_test
from thesis_code.experiments import HeldTest
from thesis_code.utils import timer, extend_seed
from thesis_code.utils import slugify


class TfModel(SequenceDist):
    def __init__(
            self,
            *,
            model_name,
            max_length,
            prompt,
            top_k,
            top_p,
            sequence_tokenizer_name=None,
            drop_prompt=True,
            device=None,
            cache_path=None,
    ):
        super().__init__()
        self._device = device or torch.device('cpu')
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self._model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if sequence_tokenizer_name is not None:
            self._sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer_name)
        else:
            self._sequence_tokenizer = None
        self._max_length = max_length
        self._prompt = prompt
        self._top_k = top_k
        self._top_p = top_p
        self._model_log_probs_cache = {}
        self._drop_prompt = drop_prompt
        self._cache_path = cache_path
        self._cache_key = model_name, max_length, prompt, top_k, top_p, sequence_tokenizer_name
        self._cached_samples = deque()
        if self._cache_path is not None:
            if os.path.exists(self._cache_path):
                cache = torch.load(self._cache_path)
            else:
                cache = {}
            if self._cache_key in cache:
                self._cached_samples = deque(cache[self._cache_key])
            else:
                cache[self._cache_key] = []
                file, filename = tempfile.mkstemp(prefix='thesis')
                os.close(file)
                torch.save(cache, filename)
                shutil.move(filename, self._cache_path)
        # with torch.no_grad():
        #     self._prompt_model_tokens = model_tokenizer(prompt, return_tensors='pt')['input_ids']

    @property
    def alphabet(self):
        return range(self._model.config.vocab_size)

    def model_tokens_to_seq(self, model_tokens):
        model_tokens = model_tokens.to('cpu')
        if self._sequence_tokenizer is None and not self._drop_prompt:
            return model_tokens.tolist()
        prompt = self._sequence_tokenizer(self._prompt)['input_ids']
        prompt_len = len(prompt)
        in_seq_token_space = self._sequence_tokenizer(
            self._model_tokenizer.decode(model_tokens, skip_special_tokens=True)
        )['input_ids']
        if self._drop_prompt:
            assert in_seq_token_space[:prompt_len] == prompt, (
                f"{in_seq_token_space[:prompt_len]=} != {prompt=}"
            )
            return in_seq_token_space[prompt_len:]
        else:
            return in_seq_token_space

    def seq_to_model_tokens(self, seq_tokens):
        if self._sequence_tokenizer is None and not self._drop_prompt:
            return torch.Tensor(seq_tokens)
        prompt = self._sequence_tokenizer(self._prompt)['input_ids']
        if self._drop_prompt:
            seq_tokens = prompt + list(seq_tokens)
        else:
            seq_tokens = list(seq_tokens)
        suffix = self._sequence_tokenizer.decode(seq_tokens, skip_special_tokens=True)
        input_ids = self._model_tokenizer(suffix, return_tensors='pt')['input_ids']
        return input_ids.to(self._device)

    def _generate_samples(self, rng, N_sequences):
        if len(self._cached_samples) >= N_sequences:
            result = [self._cached_samples.popleft() for _ in range(N_sequences)]
            return result
        from torch import ones_like, multinomial, cat
        from torch.nn.functional import softmax, log_softmax
        sequences = []
        top_k = self._top_k
        top_p = self._top_p
        # set_seed(rng.integers(2**30))
        inputs = self._model_tokenizer([self._prompt] * N_sequences, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self._device)
        for _ in range(self._max_length):
            attention_mask = ones_like(input_ids)
            model_output = self._model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = model_output.logits[:, -1, :]
            filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = softmax(filtered_next_token_logits, dim=-1)
            next_token = multinomial(probs, num_samples=1)
            # log_probs = log_softmax(filtered_next_token_logits, dim=-1)
            # print(log_probs[0, next_token[0]].item(), next_token_logits[0, next_token[0]].item())
            input_ids = cat([input_ids, next_token], dim=-1)
        for i in range(N_sequences):
            for j in range(self._max_length):
                if input_ids[i, j] == self._model_tokenizer.eos_token:
                    sequences.append(input_ids[i, :j + 1])
                    break
            else:
                sequences.append(input_ids[i, :])
        sequences = [
            self.model_tokens_to_seq(s)
            for s in sequences
        ]
        if self._cache_path is not None:
            cache = torch.load(self._cache_path)
            for seq in sequences:
                cache[self._cache_key].append(seq)
            file, filename = tempfile.mkstemp(prefix='thesis')
            os.close(file)
            torch.save(cache, filename)
            shutil.move(filename, self._cache_path)
        return sequences

    def generate_samples(self, rng, N_sequences):
        with torch.no_grad():
            result = self._generate_samples(rng, N_sequences)
        return result

    def log_p_edit_over_p_noedit(self, sequence, edit) -> float:
        kind, *args = edit
        if kind == 'replace':
            at, x = args
            if at == len(sequence) - 1:
                input_ids = self.seq_to_model_tokens(sequence)
                log_probs = self.model_log_probs(input_ids)
                return log_probs[-1, x].item() - log_probs[-1, sequence[at]].item()
        edited = apply_basic_edit(edit, sequence)
        with torch.no_grad():
            log_p_seq = self._log_p_sequence(sequence)
            log_p_edited = self._log_p_sequence(edited)
            if log_p_edited == -np.inf:
                return -np.inf
        return log_p_edited - log_p_seq

    def model_log_probs(self, input_ids):
        assert input_ids.dim() == 2, f"{input_ids.dim()=} != 2"
        assert input_ids.size(0) == 1, f"{input_ids.size(0)=} != 1"
        key = tuple(input_ids[0, :-1].tolist())
        if key in self._model_log_probs_cache:
            return self._model_log_probs_cache[key]
        prompt_len = len(self._model_tokenizer(self._prompt)['input_ids'])
        attention_mask = torch.ones_like(input_ids)
        model_output = self._model(input_ids=input_ids, attention_mask=attention_mask)
        filtered_logits = top_k_top_p_filtering(
            model_output.logits[0, prompt_len - 1:-1],
            top_k=self._top_k,
            top_p=self._top_p,
        )
        all_log_probs = nn.functional.log_softmax(filtered_logits, dim=-1)
        self._model_log_probs_cache[key] = all_log_probs
        return all_log_probs

    def _log_p_sequence(self, sequence) -> float:
        # sequence = torch.Tensor(sequence).unsqueeze(0).int()
        # input_ids = torch.cat([self._prompt_model_tokens, sequence], dim=-1)
        # attention_mask = torch.ones_like(input_ids)
        prompt_len = len(self._model_tokenizer(self._prompt)['input_ids'])
        input_ids = self.seq_to_model_tokens(sequence)
        length = input_ids.shape[1] - prompt_len
        if length > self._max_length:
            return -np.inf
        all_log_probs = self.model_log_probs(input_ids)
        seq_log_probs = all_log_probs[:, input_ids[0, prompt_len:]].diag()
        seq_log_probs.nan_to_num_(-torch.inf)
        assert torch.isfinite(seq_log_probs).all(), (
            f"not all finite: {seq_log_probs=}"
        )
        return seq_log_probs.sum().item()

    def log_p_sequence(self, sequence) -> float:
        with torch.no_grad():
            result = self._log_p_sequence(sequence)
        return result


class ConditionalTfModel(SequenceDist):
    def __init__(self, stopping_token, underlying_model):
        super().__init__()
        self._stopping_token = stopping_token
        self._model = underlying_model
        self._candidates_inspected = 0
        self._candidates_successful = 0

    def generate_samples(self, rng, N_sequences):
        sequences = []
        N_at_a_time = N_sequences * 2
        with torch.no_grad():
            while len(sequences) < N_sequences:
                candidates = self._model._generate_samples(rng, N_at_a_time)
                for candidate in candidates:
                    self._candidates_inspected += 1
                    for i in range(1, len(candidate)):
                        if candidate[i] == self._stopping_token:
                            sequences.append(candidate[:i + 1])
                            self._candidates_successful += 1
                            break
                    if len(sequences) == N_sequences:
                        break
        return sequences

    @property
    def hit_rate(self):
        if self._candidates_inspected == 0:
            return np.nan
        return self._candidates_successful / self._candidates_inspected

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        alt_seq = apply_basic_edit(edit, sequence)
        assert sequence[-1] == self._stopping_token, f"{sequence[-1]=} != {self._stopping_token=}"
        if alt_seq[-1] != sequence[-1]:
            return -np.inf
        else:
            return self._model.log_p_edit_over_p_noedit(sequence, edit)


class MixtureModel(SequenceDist):
    def __init__(self, modelA, modelB, weightA, weightB):
        super().__init__()
        self._modelA = modelA
        self._modelB = modelB
        assert np.isclose(weightA + weightB, 1)
        self._weightA = weightA
        self._weightB = weightB

    def generate_samples(self, rng, N_sequences):
        N_a = rng.binomial(N_sequences, self._weightA)
        N_b = N_sequences - N_a
        seqs_a = self._modelA.generate_samples(rng, N_a)
        seqs_b = self._modelB.generate_samples(rng, N_b)
        sequences = seqs_a + seqs_b
        rng.shuffle(sequences)
        return sequences


class TfActionSet(ActionSet):
    def __init__(
            self,
            tf_model: TfModel,
            edit_location: int,
            allow_ins_and_del: bool,
    ):
        self._tf_model = tf_model
        assert edit_location < 0, "this only works for negative edit locations, for now"
        self._edit_location = edit_location
        self._allow_ins_and_del = allow_ins_and_del

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        model_tokens = self._tf_model.seq_to_model_tokens(sequence)
        assert model_tokens[0, -1].item() == sequence[-1], (
            "this only works if model tokenizer == sequence tokenizer, "
            f"{model_tokens[0, -1].item()=} != {sequence[-1]=}, "
            f"{model_tokens=} != {sequence=}"
        )
        log_probs = self._tf_model.model_log_probs(model_tokens)
        alt_tokens = torch.where(log_probs[self._edit_location, :] > -torch.inf)[0]
        orig_token = sequence[self._edit_location]
        sequence = Sequence(sequence)
        result = dict(
            sequence.sub(len(sequence) + self._edit_location, alt_token)
            for alt_token in alt_tokens.tolist()
            if alt_token != orig_token
        )
        if self._allow_ins_and_del:
            alt_seq, edits = sequence.del_(len(sequence) + self._edit_location)
            result[alt_seq] = edits
            ins_tokens = torch.where(log_probs[self._edit_location + 1, :] > -torch.inf)[0]
            result |= dict(
                sequence.insert(len(sequence) + self._edit_location + 1, ins_token)
                for ins_token in ins_tokens.tolist()
            )
        return result


def get_point(
        *,
        sample_size: int,
        N_repeats_per_test: int,
        N_indep_tests: int,
        N_bootstrap: int,
        desired_level: float,
        max_length: int,
        perturbation: float,
        top_k: int,
        top_p: float,
        sample_size_null=None,
        perturbation_sample_size_50=None,
):
    if sample_size_null is not None:
        sample_size = sample_size_null
        perturbation = 0
    if perturbation_sample_size_50 is not None:
        sample_size = 50
        perturbation = perturbation_sample_size_50
    assert 0 <= perturbation <= 1
    null_model_prompt = "How are"
    ground_truth_prompt = "Where is"
    fast_dev_run = False
    kernel = ContiguousSubsequenceKernel(1, 1)
    ground_truth_name = null_model_name = "gpt2"
    cache_path = Path(__file__).parent.parent / 'figures' / 'exp_language' / 'cache.pt'
    inner_null_model = TfModel(
        model_name=null_model_name,
        sequence_tokenizer_name=null_model_name,
        max_length=max_length,
        prompt=null_model_prompt,
        top_k=top_k,
        top_p=top_p,
        cache_path=cache_path,
    )
    seq_tokenizer = inner_null_model._sequence_tokenizer
    qmark_token = seq_tokenizer('?', return_tensors='pt', add_special_tokens=False)['input_ids'].item()
    null_model = ConditionalTfModel(
        qmark_token,
        inner_null_model,
    )
    alternative_model = ConditionalTfModel(
        qmark_token,
        TfModel(
            model_name=ground_truth_name,
            sequence_tokenizer_name=null_model_name,
            max_length=max_length,
            prompt=ground_truth_prompt,
            top_k=top_k,
            top_p=top_p,
            cache_path=cache_path,
        ),
    )
    ground_truth = MixtureModel(
        null_model, alternative_model,
        1 - perturbation, perturbation,
    )
    action_sets = {
        "sub_only":
            TfActionSet(
                inner_null_model,
                edit_location=-2,
                allow_ins_and_del=False,
            ),
        "sub_ins_and_del'":
            TfActionSet(
                inner_null_model,
                edit_location=-2,
                allow_ins_and_del=True,
            ),
    }
    stein_ops = {
        k: ZanellaSteinOperator(
            action_set=action_set,
            model=null_model,
            kappa='barker',
        )
        for k, action_set in action_sets.items()
    }
    tests = {}
    if sample_size <= 10:
        tests |= {
            'MMD': HeldTest(lambda rng: MMDTestParametricBootstrap(
                rng=rng,
                kernel=kernel,
                target_model=null_model,
                n_mmd_samples=sample_size,
                n_bootstrap=N_bootstrap,
                desired_level=desired_level,
                sample_size=sample_size,
            )),
            'MMD(n=100)': HeldTest(lambda rng: MMDTestParametricBootstrap(
                rng=rng,
                kernel=kernel,
                target_model=null_model,
                n_mmd_samples=100,
                n_bootstrap=N_bootstrap,
                desired_level=desired_level,
                sample_size=sample_size,
            )),
            **{
                f'Stein({k})': HeldTest(lambda rng: SteinTestParametricBootstrap(
                    rng=rng,
                    kernel=kernel,
                    stein_op=stein_op,
                    target_model=null_model,
                    n_bootstrap=N_bootstrap,
                    desired_level=desired_level,
                    sample_size=sample_size,
                )) for k, stein_op in stein_ops.items()
            },
        }
    tests |= {
        'MMD-Wild(n=100)': HeldTest(lambda rng: MMDTestWildBootstrap(
            rng=rng,
            kernel=kernel,
            target_model=null_model,
            n_mmd_samples=100,
            n_bootstrap=N_bootstrap,
            desired_level=desired_level,
            sample_size=sample_size,
        )),
        **{
            f'Stein-Wild({k})': HeldTest(lambda rng: SteinTestWildBootstrap(
                rng=rng,
                kernel=kernel,
                stein_op=stein_op,
                target_model=null_model,
                n_bootstrap=N_bootstrap,
                desired_level=desired_level,
                sample_size=sample_size,
            ))
            for k, stein_op in stein_ops.items()
        },
    }

    def after_test_hook():
        print(f"{null_model.hit_rate=} {alternative_model.hit_rate=}")

    point = dict(
        tests=tests,
        sample_model=ground_truth,
        N_sequences=sample_size,
        N_repeats_per_test=N_repeats_per_test,
        N_indep_tests=N_indep_tests,
        after_test_hook=after_test_hook,
    )
    return point


def plot_results(results):
    for var_to_plot in ['rejection_rate']:
        for parameter, values in results.items():
            to_plot = {k: {k2: v2[var_to_plot] for k2, v2 in v.items()} for k, v in values.items()}
            df = pd.DataFrame(to_plot).T
            df.index.name = parameter
            filename = Path(__file__).parent.parent / f'figures/{experiment_name}_{var_to_plot}_by_{parameter}.csv'
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


experiment_name = 'exp_language'


def run_experiment(*, test_run):
    # version = '6'
    version = '7'
    baseline_parameters = {
        'max_length': 5,
        'sample_size': 10,
        'top_k': 16,
        'top_p': 1.0,
        'desired_level': 0.05,
        'perturbation': 0.5,
    }
    if test_run:
        baseline_parameters |= {
            'N_indep_tests': 1,
            'N_repeats_per_test': 1,
            'N_bootstrap': 10,
        }
    else:
        baseline_parameters |= {
            'N_indep_tests': 100,
            # 'N_indep_tests': 10,
            'N_repeats_per_test': 1,
            'N_bootstrap': 40 if version == '6' else (100 if version == '7' else None),
        }
    if test_run:
        parameters_to_vary = {
            'sample_size': [1],
        }
    else:
        parameters_to_vary = {
            # 'sample_size': [5, 10, 20, 40],
            # 'sample_size': [50, 100, 150],
            # 'sample_size_null': [50, 100, 150],
            'perturbation': np.linspace(0, 1, 9),
            'perturbation_sample_size_50': np.linspace(0, 0.5, 9),
            # 'max_length': [3, 5, 7, 10],
        }

    root_seed = 9824740821212930092259595777667619742, ''
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
                after_test_hook = point['after_test_hook']
                for test_name, test in point['tests'].items():
                    if not test_run:
                        print('', test_name)
                    test_kwargs = point | {'test': test, 'test_run': test_run}
                    del test_kwargs['tests']
                    del test_kwargs['after_test_hook']
                    value_part = slugify(str(value))
                    cache_path = Path(
                        __file__).parent.parent / f'figures/{experiment_name}/v{version}__{slugify(parameter)}__{value_part}__{slugify(test_name)}.pickle'
                    results[parameter][value][test_name] = the_results = run_test(
                        seed=extend_seed(root_seed, f'{parameter}/{value}/{test_name}'),
                        cache_path=cache_path,
                        **test_kwargs,
                    )
                    after_test_hook()
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
