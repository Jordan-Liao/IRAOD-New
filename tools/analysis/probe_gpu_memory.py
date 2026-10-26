#!/usr/bin/env python
"""Compare CUDA's reported memory with physical GPU memory on Linux.

The default check only reads memory information. An optional bounded allocation
test uses PyTorch normally and releases its allocations before exiting.
"""

import argparse
import ctypes
import ctypes.util
import datetime
import json
import os
from pathlib import Path

import torch


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ('total', ctypes.c_ulonglong),
        ('free', ctypes.c_ulonglong),
        ('used', ctypes.c_ulonglong),
    ]


class PhysicalNvml:
    """Read original NVML counters when HAMi intercepts ordinary NVML queries."""

    def __init__(self, device):
        self.library = ctypes.CDLL(
            ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
        loader = ctypes.CDLL('libdl.so.2')
        loader.dlvsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        loader.dlvsym.restype = ctypes.c_void_p
        address = loader.dlvsym(loader._handle, b'dlsym', b'GLIBC_2.2.5')
        if not address:
            raise RuntimeError('Original NVML symbol lookup is unavailable')
        resolve = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(address)

        def function(name, *argtypes):
            address = resolve(self.library._handle, name.encode())
            if not address:
                raise RuntimeError(f'NVML symbol unavailable: {name}')
            return ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(address)

        self.initialize = function('nvmlInit_v2')
        self.shutdown = function('nvmlShutdown')
        self.get_handle = function(
            'nvmlDeviceGetHandleByUUID', ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p))
        self.get_memory = function(
            'nvmlDeviceGetMemoryInfo', ctypes.c_void_p,
            ctypes.POINTER(NvmlMemory))
        self._check(self.initialize(), 'nvmlInit_v2')
        try:
            visible_handle = ctypes.c_void_p()
            visible_devices = self.library.nvmlDeviceGetHandleByIndex_v2
            visible_devices.argtypes = [
                ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            visible_devices.restype = ctypes.c_int
            self._check(visible_devices(device, ctypes.byref(visible_handle)),
                        'nvmlDeviceGetHandleByIndex_v2')
            get_uuid = self.library.nvmlDeviceGetUUID
            get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            get_uuid.restype = ctypes.c_int
            uuid_buffer = ctypes.create_string_buffer(96)
            self._check(get_uuid(visible_handle, uuid_buffer, len(uuid_buffer)),
                        'nvmlDeviceGetUUID')
            self.uuid = uuid_buffer.value.decode()
            self.handle = ctypes.c_void_p()
            self._check(self.get_handle(uuid_buffer.value, ctypes.byref(self.handle)),
                        'nvmlDeviceGetHandleByUUID')
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _check(result, operation):
        if result:
            raise RuntimeError(f'{operation} failed with NVML status {result}')

    def snapshot(self):
        memory = NvmlMemory()
        self._check(self.get_memory(self.handle, ctypes.byref(memory)),
                    'nvmlDeviceGetMemoryInfo')
        return {name + '_bytes': getattr(memory, name)
                for name in ('total', 'free', 'used')}


def display_memory(label, memory):
    values = {key.replace('_bytes', '_gib'): round(value / 2**30, 3)
              for key, value in memory.items()}
    print(f'{label}: {json.dumps(values)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0,
                        help='NVML visible device index (default: 0)')
    parser.add_argument('--allocate-gib', type=float, default=0,
                        help='Opt into a bounded allocation test (default: 0)')
    parser.add_argument('--chunk-mib', type=int, default=256)
    parser.add_argument('--output', type=Path, help='Save a JSON diagnosis')
    args = parser.parse_args()
    if args.device < 0 or args.allocate_gib < 0 or args.chunk_mib <= 0:
        parser.error('device and allocate-gib must be nonnegative; chunk-mib must be positive')
    if not torch.cuda.is_available():
        parser.exit(1, 'CUDA is unavailable in this Python environment.\n')

    report = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'torch_version': torch.__version__,
        'virtual_gpu_settings': {
            name: value for name, value in os.environ.items()
            if name.startswith('CUDA_DEVICE_')},
        'requested_allocation_bytes': int(args.allocate_gib * 2**30),
    }
    driver = None
    chunks = []
    exit_code = 0
    try:
        try:
            driver = PhysicalNvml(args.device)
            report['gpu_uuid'] = driver.uuid
            report['physical_before_cuda'] = driver.snapshot()
            display_memory('Physical GPU before CUDA initialization',
                           report['physical_before_cuda'])
        except (OSError, RuntimeError, AttributeError) as error:
            report['physical_query_error'] = str(error)
            print(f'Physical GPU memory query unavailable: {error}', flush=True)

        report['device_name'] = torch.cuda.get_device_name(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        report['cuda_reported'] = {'total_bytes': total, 'free_bytes': free}
        print(f"Device: {report['device_name']}", flush=True)
        display_memory('CUDA reported GPU memory', report['cuda_reported'])
        chunk_bytes = args.chunk_mib * 2**20
        allocated_bytes = 0
        while allocated_bytes < report['requested_allocation_bytes']:
            size = min(chunk_bytes,
                       report['requested_allocation_bytes'] - allocated_bytes)
            chunks.append(torch.empty(size, dtype=torch.uint8,
                                      device=f'cuda:{args.device}'))
            torch.cuda.synchronize(args.device)
            allocated_bytes += size
            report['successful_allocation_bytes'] = allocated_bytes
            print(f'Allocated {allocated_bytes / 2**30:.3f} GiB', flush=True)
        if args.allocate_gib:
            report['allocation_status'] = 'passed'
    except torch.cuda.OutOfMemoryError as error:
        report['allocation_status'] = 'failed'
        report['allocation_error'] = str(error)
        exit_code = 2
        print(f'Allocation failed: {error}', flush=True)
        if driver is not None:
            report['physical_at_failure'] = driver.snapshot()
            display_memory('Physical GPU at allocation failure',
                           report['physical_at_failure'])
    finally:
        if args.allocate_gib:
            chunks.clear()
            torch.cuda.empty_cache()
            print('Probe allocations released.', flush=True)
        if driver is not None:
            driver.shutdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'Diagnosis saved to {args.output}', flush=True)

    physical = report.get('physical_before_cuda')
    reported = report.get('cuda_reported')
    if physical and reported and reported['free_bytes'] > physical['free_bytes'] + 2**30:
        print('Reported free memory exceeds physical free memory. '
              'Ask the GPU provider to check shared GPU occupancy and allocation.',
              flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
