# ds4 multi-GPU deployment on the Mac Pro 7,1

Follows on from `MACPRO_AMD_PORT.md` (build + single-GPU port). This note covers
what multi-GPU machinery DwarfStar already has, what actually works on this
machine's three AMD Metal devices, what was measured, and why the GPU still
loses to the CPU for this checkpoint.

## Hardware

| | |
|---|---|
| Metal devices | `[0]` W6800X Duo, `[1]` W6800X Duo, `[2]` Vega II — 32 GiB each, **96 GiB total** |
| Model | DeepSeek-V4-Flash IQ2XXS/Q2_K, 80.76 GiB |
| Unified memory | no — every device is discrete |

The model fits in the combined VRAM but not in any single device, so multi-GPU
is the only way the GPU path could avoid host-memory weight reads.

## What ds4 already provides

There are two, independent multi-GPU designs:

1. **In-process multi-tier** (`--gpu-devices` / `--gpu-vram`, layer placement,
   per-device tensor caches, cross-device copies, tensor-parallel reductions).
   This is **CUDA-only**. On Apple, `ds4_gpu_init_multi`,
   `ds4_gpu_tensor_alloc_on`, `ds4_gpu_tensor_copy_xdev`,
   `ds4_gpu_lookup_cache_strict` and `ds4_gpu_device_cache_tensors` are stubs in
   `ds4.c`, and `ds4_metal.m` has no device concept at all — the backend is a
   single global `g_device`/`g_queue`. The placement code is backend-agnostic,
   but exercising it needs a Metal implementation of the whole mgpu surface.

2. **Network distributed modes** (TP and PP), which each run as **one process
   per rank**. These work with Metal and were used here.

### Device selection (added)

The Metal backend always used `MTLCreateSystemDefaultDevice()`. It now honours:

```
DS4_METAL_DEVICE_INDEX=N      index into MTLCopyAllDevices()
DS4_METAL_DEVICE_NAME=SUBSTR  case-insensitive name match (stable across reboots)
```

With neither set, behaviour is unchanged. When set, the device list is printed
with indices so the next process can be pinned.

### One process per rank needs a distinct lock

`ds4` refuses to start a second instance via `flock` on `/tmp/ds4.lock`. Set a
per-process path when deploying more than one rank on the host:

```
DS4_LOCK_FILE=/tmp/ds4.lock.$N
```

## Recipes that run on this machine

Tensor parallelism (2 ranks, TCP on loopback; workers start first and retry):

```sh
DS4_LOCK_FILE=/tmp/ds4.w1.lock DS4_METAL_DEVICE_INDEX=1 ./ds4 \
  --tensor-parallel --role worker --coordinator 127.0.0.1 9911 --transport tcp

DS4_LOCK_FILE=/tmp/ds4.c0.lock DS4_METAL_DEVICE_INDEX=0 ./ds4 \
  --tensor-parallel --role coordinator --listen 127.0.0.1 9911 --transport tcp \
  -p "..." -n 16
```

Pipeline parallelism (N stages; `--layers` is inclusive, `N:output` takes the head):

```sh
DS4_LOCK_FILE=/tmp/ds4.w1.lock DS4_METAL_DEVICE_INDEX=1 ./ds4 \
  --role worker --layers 14:28 --coordinator 127.0.0.1 9922
DS4_LOCK_FILE=/tmp/ds4.w2.lock DS4_METAL_DEVICE_INDEX=2 ./ds4 \
  --role worker --layers 29:output --coordinator 127.0.0.1 9922
DS4_LOCK_FILE=/tmp/ds4.c0.lock DS4_METAL_DEVICE_INDEX=0 ./ds4 \
  --role coordinator --layers 0:13 --listen 127.0.0.1 9922 -p "..." -n 16
```

Note: worker `--listen` rejects port `0`; omit the flag and the worker picks an
ephemeral data port. TP is hard-limited to one 50/50 worker.

## Measurements (greedy, same model/prompt)

| Configuration | Prefill | Generation | Output |
|---|---:|---:|---|
| 1 GPU (W6800X) | 0.5 t/s | 0.19–0.23 t/s | coherent |
| 2 GPUs, tensor parallel | 0.14 t/s | 0.20 t/s | coherent |
| 2 GPUs, pipeline (0:21 / 22:output) | 0.42 t/s | 0.33 t/s | coherent |
| 3 GPUs, pipeline (14/15/14 layers) | 0.40 t/s | 0.40 t/s | **incoherent** |
| CPU, 28 threads | 3.09 t/s | 2.69 t/s | coherent |

* Context must be the default (32768). With `-c 4096` even the single-GPU
  backend emits gibberish, so a small context is not a multi-GPU question.
* Tensor parallelism maps 219 spans / 44.48 GiB per rank (a 50/50 expert shard)
  and halves each rank's weight bytes, but generation is unchanged: the fixed
  per-command-buffer cost dominates.
* The **2-stage** pipeline is the best GPU configuration measured: coherent and
  ~1.5× single-GPU generation, because stages overlap across tokens (the
  coordinator reports `receive prefetch depth 2`; a throughput effect, not a
  latency one).
* The **3-stage** pipeline runs and is fastest, but produced incoherent text at
  both `-c 4096` and the default context, with two different layer splits. The
  first split (`0:13 / 14:28 / 29:output`) and a second (`0:21 / 22:32 /
  33:output`) both failed while the 2-stage split with the same boundary
  (`0:21 / 22:output`) was coherent. That points at intermediate-worker
  forwarding (multi-hop routes), not the layer boundaries or the SIMD-group
  shim — it is not usable until that is diagnosed.

## Why splitting work across GPUs does not help enough

`DS4_METAL_CB_TIMES=1` on one GPU, per decode token:

```
encode           0.5–0.9 s     (CPU building the command buffer)
gpu span         2.15 s        (GPU actually busy)
commit -> done   4.5–5.3 s     (of which ~3 s is driver side)
```

The prefill command buffer had a 25.7 s GPU span. Two conclusions:

* The 2.15 s GPU span is host-memory (PCIe) bound — the weights are no-copy
  shared views.
* The ~3 s driver overhead is **per command buffer** and does not shrink when
  the work moves to another GPU. `DS4_METAL_MODEL_UNTRACKED=1` (disable hazard
  tracking) changed nothing, so it is not hazard tracking; it is consistent with
  the driver validating/residing the large shared views, which is exactly what
  the (unavailable on AMD) residency set was meant to avoid.

So TP halves the bytes but not the fixed cost, and PP only wins by overlapping
independent tokens across stages.

## The real fix: weights resident in VRAM — and where it hits the wall

I prototyped an opt-in `DS4_METAL_MODEL_VRAM` that made each model view a
`MTLStorageModePrivate` copy instead of a no-copy shared view (and merged
overlapping spans so no byte was copied twice). Result for the 14:28 slice:

```
restricting metal model map to layers 14:28 (31 spans, 27.67 GiB tensor span)
VRAM copy would exceed the device working set (31.98 + 0.14 > 31.98 GiB)
```

The union is 27.67 GiB, but Metal caps one buffer at `maxBufferLength`
(3.5 GiB), so the runtime splits the slice into overlapping views whose copies
total ~32.1 GiB — just over the 32 GiB working set. The overlap exists so no
weight tensor is split across two buffers; the largest tensor in the slice plus
the view count sets the overhead.

That experiment was reverted (it cannot succeed for this model and was not
validated end to end). A per-**tensor** device cache would remove the overlap:
this is what the CUDA `ds4_gpu_device_cache_tensors` /
`ds4_gpu_lookup_cache_strict` pair already does, and porting that pair to Metal
— plus making the Metal dispatch prefer a device-resident range — is the
concrete work item. The engine already computes the per-device tensor ranges and
calls these hooks; only the Metal implementations are missing.

## Recommendation

* **Interactive single-stream use on this machine: CPU backend.** It is ~8×
  faster than the best coherent GPU configuration (2.69 vs 0.33 t/s) and needs
  one process.
* **Multi-GPU is a capacity/throughput tool here, not a latency tool.** The
  coherent 2-stage pipeline gave ~1.5× single-GPU generation by overlapping
  tokens; that helps long prompts and concurrent sessions, not one chat. The
  faster 3-stage route must not be used until its incoherent output is fixed.
* If GPU acceleration for a single stream is wanted, the work is the Metal
  per-device VRAM cache above (CUDA mgpu API port), not more ranks. Even then,
  a pipeline split runs stages in sequence, so expect it to approach — not
  dwarf — the CPU.
* Two enabling changes should be upstreamable on their own:
  `DS4_METAL_DEVICE_INDEX`/`DS4_METAL_DEVICE_NAME`, and documenting
  `DS4_LOCK_FILE` for same-host multi-rank deployments.
