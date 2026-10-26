# GPU 显存诊断（2026-10-01）

## 结论

这次 CUDA OOM 的直接原因是底层物理 GPU 显存不足。容器的 40GB 是 HAMi
虚拟显存配额，无法说明底层 GPU 当前还剩多少可用物理显存。

GPU UUID：`GPU-35ced431-b8d6-6ed6-638f-26b0dce4385e`。

`/etc/ld.so.preload` 加载了 `/usr/local/vgpu/libvgpu.so`，实例使用 HAMi
软件 vGPU。`nvidia-smi` 显示 `MIG mode: Disabled`，不是 MIG 硬件分区。
因此，容器显示的空闲配额和底层 GPU 当前的物理空闲显存必须分别核对。

读取原始 NVIDIA NVML 驱动的结果：

| 指标 | 诊断实测 |
| --- | ---: |
| 物理显存总容量 | 79.647 GiB |
| 测试前物理显存已用 | 76.258 GiB |
| 测试前物理显存空闲 | 3.390 GiB |
| CUDA 显示的虚拟显存总容量 | 39.823 GiB |
| CUDA 显示的虚拟空闲显存 | 39.384 GiB |

独立测试不加载模型或数据，只通过 PyTorch 逐次申请 256 MiB 显存。
成功分配 2.750 GiB 后，再申请 256 MiB 就报 OOM。此时物理 GPU 仅剩
0.195 GiB，而 CUDA 错误信息仍声称剩余 36.63 GiB。测试结束后已释放分配。

随后再次进行了默认的只读检查，没有申请测试显存，得到相同的测试前物理
空闲容量 3.390 GiB。该次机器可读报告保存在
`deploy_env/gpu_memory_snapshot_20261001.json`。

2026-10-01 的 `deploy_env/mmcv_h100_fix_20261001.wrS1HO/project_smoke_test.log`
也记录了相同问题：模型只分配约 2.20 GiB，再申请 40 MiB 失败，但报错显示
剩余 36.60 GiB。普通 `nvidia-smi` 的容器进程列表为空，无法据此排除共享 GPU
上的其他占用。

## 如何修复

需要 GPU 服务平台检查该 GPU 上的共享占用和调度配额，释放物理显存或迁移
到有足够实际可用显存的实例。当前容器无法管理其他实例的显存占用。
具体是调度超额分配、资源占用未释放还是平台管理状态异常，需要平台核查。
修改 batch size、清理 PyTorch 缓存或设置碎片化参数不会恢复承诺的 40GB
实际可用容量。平台修复后，用下面的检查确认，再重跑原训练配置。

## 重现命令

默认只读取显存信息：

```bash
/opt/conda/envs/iraod/bin/python tools/analysis/probe_gpu_memory.py
```

主动进行最多 4 GiB 的分配测试，并保存报告（失败时退出码为 2）：

```bash
/opt/conda/envs/iraod/bin/python tools/analysis/probe_gpu_memory.py \
  --allocate-gib 4 \
  --output deploy_env/gpu_memory_probe_20261001.json
```

诊断通过原始 NVML 只读接口获取物理显存统计；所有测试分配仍经过原来的
PyTorch/CUDA/HAMi 路径，不修改或绕过平台配额。

## 可转发给平台的说明

> 当前实例显示约 40GB 显存，但 CUDA 实际只能分配约 2.75 GiB。底层 GPU
> UUID 为 GPU-35ced431-b8d6-6ed6-638f-26b0dce4385e，物理总容量 79.647 GiB，
> 测试前已使用 76.258 GiB、仅余 3.390 GiB。独立 PyTorch 分配测试可复现，
> 未运行模型。请检查该 GPU 的共享占用、HAMi 配额及资源调度，释放显存或迁移实例。
