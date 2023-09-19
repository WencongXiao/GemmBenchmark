# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

r'''GEMM benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import math
import numpy as np
import pynvml
import sys
import time
import torch

from llmacc.tools import utils


GemmBenchmarkResult = collections.namedtuple(
    'GemmBenchmarkResult', [
        'title', 'gap_secs',
        'time_all_ms',
        'clock_sm_all_mhz', 'clock_sm_limit_mhz',
        'clock_mem_all_mhz', 'clock_mem_limit_mhz',
        'power_all_w', 'power_limit_w',
        'temp_all_c', 'temp_limit_c'])


GemmBenchmarkStats = collections.namedtuple(
    'GemmBenchmarkStats', [
        'time_median_ms', 'time_jitter_ms',
        'speed_avg_tflops', 'speed_min_tflops', 'speed_max_tflops',
        'clock_sm_avg_mhz', 'clock_sm_min_mhz', 'clock_sm_max_mhz',
        'clock_mem_avg_mhz', 'clock_mem_min_mhz', 'clock_mem_max_mhz',
        'power_avg_w', 'power_min_w', 'power_max_w',
        'temp_avg_c', 'temp_min_c', 'temp_max_c'])


class GemmBenchmark:
    def __init__(self, num_epochs, epoch_steps, step_repeats):
        self.num_epochs = num_epochs
        self.epoch_steps = epoch_steps
        self.step_repeats = step_repeats
        self.gloo_world = torch.distributed.new_group(
            list(range(torch.distributed.get_world_size())), backend='gloo')
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device_count = torch.cuda.device_count()
        self.node_rank = self.rank // self.device_count
        self.device = self.rank % self.device_count

        pynvml.nvmlInit()
        self.nvml = pynvml.nvmlDeviceGetHandleByIndex(self.device)

    @torch.no_grad()
    def __call__(self, lmat, rmat, out, name, gap_secs=None):
        if self.num_epochs < 1 or self.epoch_steps < 1 or self.step_repeats < 1:
            return None, None

        w, s1 = list(lmat.shape)
        s2, h = list(rmat.shape)
        if s1 != s2:
            raise ValueError(
                'Shape of two tensor is not compatible: '
                f'({w}, {s1}) x ({s2}, {h})')
        flos = 2 * w * s1 * h
        gap_str = '' if gap_secs is None else f' with gap {gap_secs:.3f}s'
        title = f'GEMM {name} benchmark{gap_str}'
        torch.cuda.synchronize(device=self.device)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}] '
              f'[{self.node_rank}:{self.device}] '
              f'{title} on device {self.device} | started...\n',
              file=sys.stderr,
              flush=True)

        for _ in range(self.step_repeats):
            torch.matmul(lmat, rmat, out=out)
        
        clock_sm_limit_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
            self.nvml, pynvml.NVML_CLOCK_SM)
        clock_mem_limit_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
            self.nvml, pynvml.NVML_CLOCK_MEM)
        power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.nvml) * 1e-3
        temp_limit_c = pynvml.nvmlDeviceGetTemperatureThreshold(
            self.nvml, pynvml.NVML_TEMPERATURE_GPU)
        time_all_ms = []
        clock_sm_all_mhz = []
        clock_mem_all_mhz = []
        power_all_w = []
        temp_all_c = []
        for epoch_idx in range(self.num_epochs):
            time_epoch_ms = []
            clock_sm_epoch_mhz = []
            clock_mem_epoch_mhz = []
            power_epoch_w = []
            temp_epoch_c = []
            for _ in range(self.epoch_steps):
                torch.cuda.synchronize(device=self.device)
                start = time.perf_counter()
                for _ in range(self.step_repeats):
                    torch.matmul(lmat, rmat, out=out)
                torch.cuda.synchronize(device=self.device)
                end = time.perf_counter()
                clock_sm_mhz = pynvml.nvmlDeviceGetClockInfo(
                    self.nvml, pynvml.NVML_CLOCK_SM)
                clock_mem_mhz = pynvml.nvmlDeviceGetClockInfo(
                    self.nvml, pynvml.NVML_CLOCK_MEM)
                power_w = pynvml.nvmlDeviceGetPowerUsage(self.nvml) * 1e-3
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    self.nvml, pynvml.NVML_TEMPERATURE_GPU)
                if gap_secs is not None:
                    time.sleep(gap_secs)
                step_time_ms = 1.e3 * (end - start) / self.step_repeats
                time_epoch_ms.append(step_time_ms)
                clock_sm_epoch_mhz.append(clock_sm_mhz)
                clock_mem_epoch_mhz.append(clock_mem_mhz)
                power_epoch_w.append(power_w)
                temp_epoch_c.append(temp_c)
            time_all_ms.extend(time_epoch_ms)
            clock_sm_all_mhz.extend(clock_sm_epoch_mhz)
            clock_mem_all_mhz.extend(clock_mem_epoch_mhz)
            power_all_w.extend(power_epoch_w)
            temp_all_c.extend(temp_epoch_c)
            if self.num_epochs <= 1:
                continue
            epoch_time_avg_ms = np.mean(time_epoch_ms)
            epoch_time_min_ms = np.min(time_epoch_ms)
            epoch_time_max_ms = np.max(time_epoch_ms)
            epoch_time_median_ms = np.median(time_epoch_ms)
            epoch_time_jitter_ms = utils.compute_jitter(time_epoch_ms)
            epoch_speed_avg_tflops = flos / epoch_time_avg_ms * 1e-9
            epoch_speed_min_tflops = flos / epoch_time_max_ms * 1e-9
            epoch_speed_max_tflops = flos / epoch_time_min_ms * 1e-9
            epoch_clock_sm_avg_mhz = np.mean(clock_sm_epoch_mhz)
            epoch_clock_sm_min_mhz = np.min(clock_sm_epoch_mhz)
            epoch_clock_sm_max_mhz = np.max(clock_sm_epoch_mhz)
            epoch_clock_mem_avg_mhz = np.mean(clock_mem_epoch_mhz)
            epoch_clock_mem_min_mhz = np.min(clock_mem_epoch_mhz)
            epoch_clock_mem_max_mhz = np.max(clock_mem_epoch_mhz)
            epoch_power_avg_w = np.mean(power_epoch_w)
            epoch_power_min_w = np.min(power_epoch_w)
            epoch_power_max_w = np.max(power_epoch_w)
            epoch_temp_avg_c = np.mean(temp_epoch_c)
            epoch_temp_min_c = np.min(temp_epoch_c)
            epoch_temp_max_c = np.max(temp_epoch_c)
            torch.cuda.synchronize(device=self.device)
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}] '
                  f'[{self.node_rank}:{self.device}] '
                  f'{title} on device {self.device} | '
                  f'finished epoch: {epoch_idx} / {self.num_epochs} '
                  f'({self.epoch_steps * self.step_repeats} runs) | '
                  f'time: {epoch_time_median_ms:.3f}'
                  f'+/-{epoch_time_jitter_ms:.3f}ms '
                  f'({epoch_time_min_ms:.3f}ms - '
                  f'{epoch_time_max_ms:.3f}ms) | '
                  f'speed: {epoch_speed_avg_tflops:.3f}TFLOP/s '
                  f'({epoch_speed_min_tflops:.3f}TFLOP/s - '
                  f'{epoch_speed_max_tflops:.3f}TFLOP/s) | '
                  f'clock (SM): {epoch_clock_sm_avg_mhz:.3f}MHz / '
                  f'{clock_sm_limit_mhz:.3f}MHz '
                  f'({epoch_clock_sm_min_mhz:.3f}MHz - '
                  f'{epoch_clock_sm_max_mhz:.3f}MHz) | '
                  f'clock (MEM): {epoch_clock_mem_avg_mhz:.3f}MHz / '
                  f'{clock_mem_limit_mhz:.3f}MHz '
                  f'({epoch_clock_mem_min_mhz:.3f}MHz - '
                  f'{epoch_clock_mem_max_mhz:.3f}MHz) | '
                  f'power: {epoch_power_avg_w:.3f}W / '
                  f'{power_limit_w:.3f}W '
                  f'({epoch_power_min_w:.3f}W - '
                  f'{epoch_power_max_w:.3f}W) | '
                  f'temperature: {epoch_temp_avg_c:.0f}C / '
                  f'{temp_limit_c:.0f}C '
                  f'({epoch_temp_min_c:.0f}C - '
                  f'{epoch_temp_max_c:.0f}C)\n',
                  file=sys.stderr,
                  flush=True)

        time_avg_ms = np.mean(time_all_ms)
        time_min_ms = np.min(time_all_ms)
        time_max_ms = np.max(time_all_ms)
        time_median_ms = np.median(time_all_ms)
        time_jitter_ms = utils.compute_jitter(time_all_ms)
        speed_avg_tflops = flos / time_avg_ms * 1e-9
        speed_min_tflops = flos / time_max_ms * 1e-9
        speed_max_tflops = flos / time_min_ms * 1e-9
        clock_sm_avg_mhz = np.mean(clock_sm_all_mhz)
        clock_sm_min_mhz = np.min(clock_sm_all_mhz)
        clock_sm_max_mhz = np.max(clock_sm_all_mhz)
        clock_mem_avg_mhz = np.mean(clock_mem_all_mhz)
        clock_mem_min_mhz = np.min(clock_mem_all_mhz)
        clock_mem_max_mhz = np.max(clock_mem_all_mhz)
        power_avg_w = np.mean(power_all_w)
        power_min_w = np.min(power_all_w)
        power_max_w = np.max(power_all_w)
        temp_avg_c = np.mean(temp_all_c)
        temp_min_c = np.min(temp_all_c)
        temp_max_c = np.max(temp_all_c)
        torch.cuda.synchronize(device=self.device)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}] '
            f'[{self.node_rank}:{self.device}] '
            f'{title} on device {self.device} | finished all epochs '
            f'({self.num_epochs * self.epoch_steps * self.step_repeats} runs)'
            f' | time: {time_median_ms:.3f}+/-{time_jitter_ms:.3f}ms '
            f'({time_min_ms:.3f}ms - {time_max_ms:.3f}ms) | '
            f'speed: {speed_avg_tflops:.3f}TFLOP/s '
            f'({speed_min_tflops:.3f}TFLOP/s - '
            f'{speed_max_tflops:.3f}TFLOP/s) | '
            f'clock (SM): {clock_sm_avg_mhz:.3f}MHz / '
            f'{clock_sm_limit_mhz:.3f}MHz '
            f'({clock_sm_min_mhz:.3f}MHz - {clock_sm_max_mhz:.3f}MHz) | '
            f'clock (MEM): {clock_mem_avg_mhz:.3f}MHz / '
            f'{clock_mem_limit_mhz:.3f}MHz '
            f'({clock_mem_min_mhz:.3f}MHz - {clock_mem_max_mhz:.3f}MHz) | '
            f'power: {power_avg_w:.3f}W / {power_limit_w:.3f}W '
            f'({power_min_w:.3f}W - {power_max_w:.3f}W) | '
            f'temperature: {temp_avg_c:.0f}C / {temp_limit_c:.0f}C '
            f'({temp_min_c:.0f}C - {temp_max_c:.0f}C)\n',
            file=sys.stderr,
            flush=True)

        result = GemmBenchmarkResult(
            title, gap_secs, time_all_ms,
            clock_sm_all_mhz, clock_sm_limit_mhz,
            clock_mem_all_mhz, clock_mem_limit_mhz,
            power_all_w, power_limit_w,
            temp_all_c, temp_limit_c)
        stats = GemmBenchmarkStats(
            time_median_ms, time_jitter_ms,
            speed_avg_tflops, speed_min_tflops, speed_max_tflops,
            clock_sm_avg_mhz, clock_sm_min_mhz, clock_sm_max_mhz,
            clock_mem_avg_mhz, clock_mem_min_mhz, clock_mem_max_mhz,
            power_avg_w, power_min_w, power_max_w,
            temp_avg_c, temp_min_c, temp_max_c)
        return result, stats

    @torch.no_grad()
    def print_report(self, result, stats):
        if result is None or stats is None:
            return

        local_stats_tensor = torch.tensor(
            list(stats), dtype=torch.float32).cpu()
        global_stats_tensor = [
            torch.empty_like(local_stats_tensor).cpu()
            for _ in range(self.world_size)]
        torch.distributed.all_gather(
            global_stats_tensor, local_stats_tensor, group=self.gloo_world)
        all_stats = [
            GemmBenchmarkStats(*f.tolist()) for f in global_stats_tensor]
        sorted_stats = sorted(
            enumerate(all_stats), key=lambda x: x[1].speed_avg_tflops)
        all_time_ms_list = [x.time_median_ms for x in all_stats]
        all_time_ms = np.median(all_time_ms_list)
        all_time_jitter_ms = utils.compute_jitter(all_time_ms_list)
        all_min_tflops_list = [x.speed_min_tflops for x in all_stats]
        all_tflops_min = np.min(all_min_tflops_list)
        all_max_tflops_list = [x.speed_max_tflops for x in all_stats]
        all_tflops_max = np.max(all_max_tflops_list)

        if self.device != 0:
            return

        report = f'\n# {result.title} for {self.world_size} devices\n'
        report += (
            f'- clock (SM) limit: {result.clock_sm_limit_mhz:.3f}MHz\n'
            f'- clock (MEM) limit: {result.clock_mem_limit_mhz:.3f}MHz\n'
            f'- power limit: {result.power_limit_w:.3f}W\n'
            f'- temperature threshold: {result.temp_limit_c:.0f}C\n'
            f'- time: {all_time_ms:.3f}+/-{all_time_jitter_ms:.3f}ms\n'
            f'- speed: {all_tflops_min:.3f}TFLOP/s '
            f'- {all_tflops_max:.3f}TFLOP/s\n')
        report += (
            f'{"node":<6} | {"device":<6} | '
            f'{"time - ms":<18} | '
            f'{"jitter - ms":<18} | '
            f'{"speed - TFLOP/s":<18} | '
            f'{"clock (SM) - MHz":<18} | '
            f'{"clock (MEM) - MHz":<18} | '
            f'{"power - W":<18} | '
            f'{"temperature - C":<18}\n'
            f'{"-" * 6} | {"-" * 6} | '
            f'{"-" * 18} | {"-" * 18} | '
            f'{"-" * 18} | '
            f'{"-" * 18} | '
            f'{"-" * 18} | '
            f'{"-" * 18} | '
            f'{"-" * 18}\n')
        for r, s in sorted_stats:
            report += (
                f'{f"{r // self.device_count}":<6} | '
                f'{f"{r % self.device_count}":<6} | '
                f'{f"{s.time_median_ms:.3f}":<18} | '
                f'{f"{s.time_jitter_ms:.3f}":<18} | '
                f'{f"{s.speed_avg_tflops:.3f}":<18} | '
                f'{f"{s.clock_sm_avg_mhz:.3f}":<18} | '
                f'{f"{s.clock_mem_avg_mhz:.3f}":<18} | '
                f'{f"{s.power_avg_w:.3f}":<18} | '
                f'{f"{s.temp_avg_c:.0f}":<18}\n' )
        print(f'{report}\n', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        help='Collective communication backend',
        default='nccl')
    parser.add_argument(
        '--dtype',
        help='Data type of input',
        default='bfloat16')
    parser.add_argument(
        '--seqlen',
        help='Sequence length of input',
        default='4K')
    parser.add_argument(
        '--flos',
        nargs='+',
        help='Workload FLOPs',
        default=['2T', '5T', '10T'])
    parser.add_argument(
        '--gap-secs',
        type=float,
        nargs='+',
        help='Gap secs between 2 steps',
        default=[None])
    parser.add_argument('--step-repeats', type=int, default=10)
    parser.add_argument('--epoch-steps', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()
    utils.seed_everything(0)

    if args.num_epochs > 0 and args.epoch_steps >0 and args.step_repeats > 0:
        torch.distributed.init_process_group(backend=args.backend)
        benchmark = GemmBenchmark(
            args.num_epochs, args.epoch_steps, args.step_repeats)
        reports = {}
        for flos in args.flos:
            dtype = eval(f'torch.{args.dtype}')
            flos_num = utils.parse_count(flos)
            seqlen = utils.parse_count(args.seqlen)
            hidden_size = int(math.sqrt(1. * flos_num / 2. / 4. / seqlen))
            lmat = torch.randn(
                4 * hidden_size, seqlen, dtype=dtype).to(benchmark.device)
            rmat = torch.randn(
                seqlen, hidden_size, dtype=dtype).to(benchmark.device)
            out = torch.empty(
                4 * hidden_size, hidden_size, dtype=dtype).to(benchmark.device)
            reports[flos] = {}
            for gap_secs in args.gap_secs:
                gap_millis = int(gap_secs * 1e3) if gap_secs else 0
                result, stats = benchmark(
                    lmat, rmat, out,
                    f'{flos}FLOPs',
                    gap_secs=gap_secs)
                reports[flos][gap_millis] = (result, stats)
            del lmat, rmat, out
        if not args.no_report:
            for flos in args.flos:
                for gap_secs in args.gap_secs:
                    gap_millis = int(gap_secs * 1e3) if gap_secs else 0
                    benchmark.print_report(*reports[flos][gap_millis])
