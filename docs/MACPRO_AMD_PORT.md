# Running DwarfStar (ds4) on the Intel Mac Pro with AMD GPUs

Status: **ds4 builds and runs on this machine.** Inference is coherent on both the
AMD Metal path and the CPU path. For this model on this hardware the CPU backend
is the useful configuration; the GPU path works but is limited by the discrete
GPU / PCIe architecture (details below).

## Machine

| | |
|---|---|
| Host | `MacPro71.local`, Mac Pro 7,1 |
| CPU | Intel Xeon W-3275, 28 cores |
| RAM | 256 GiB |
| OS | macOS 26.6.2 (25G83), x86_64 |
| GPU | 2× AMD Radeon PRO W6800X Duo (32 GiB each die), 1× AMD Radeon Pro Vega II (32 GiB) |
| Metal | Metal 3 (`MTLGPUFamilyMetal3`), **no Metal 4**, **no unified memory** |
| `maxBufferLength` | 3.5 GiB per buffer |
| Model | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (80.76 GiB) |

Checkout: `antirez/ds4` (DwarfStar) at `ds4/`, commit `8db1d1d`.

## Why it did not build or run

Three independent problems, all fixed:

1. **Build flag.** The Makefile used `-mcpu=native` on every Darwin host. Apple's
   clang only accepts `-mcpu` for arm64; on x86_64 it fails with
   `unsupported option '-mcpu=' for target 'x86_64-apple-darwin25.6.0'`.

2. **AMD Metal cannot lower `simdgroup_matrix`.** The Metal 3 compiler on this
   AMD driver accepts the MSL source but fails at pipeline creation for every
   kernel that uses the hardware matrix builtins:

   ```
   SC compilation failure
   There is a call to an undefined label
   ```

   `ds4_gpu_init` hard-requires three GLM pipelines that use them, so the whole
   Metal backend refused to initialise. (llama.cpp hits the same limitation on
   these GPUs and works around it with a `use_mm_manual` GEMM path; ds4 had no
   fallback.)

3. **Residency sets are unsupported.** `[MTLDevice newResidencySetWithDescriptor:]`
   returns `nil` (with no error) on these AMD cards. ds4 treated that as a fatal
   model-mapping failure, even though the residency set is only a scheduling
   hint.

## Changes

### `Makefile`

`NATIVE_CPU_FLAG` now depends on the host architecture: `-march=native` on
x86_64 Darwin, `-mcpu=native` on arm64 Darwin (and `-march=native` elsewhere).

### `metal/simdgroup_shim.metal` (new)

A portable implementation of the small `simdgroup_matrix` surface DwarfStar uses:
`simdgroup_matrix`/`simdgroup_{float,half}8x8`, `make_filled_simdgroup_matrix`,
`simdgroup_load`, `simdgroup_store`, `simdgroup_multiply`, and
`simdgroup_multiply_accumulate`. It is built from `simd_shuffle` and a prefix-sum
lane id, which AMD Metal 3 **does** support. The 8×8 tile is distributed two
elements per lane; the memory addressing keeps Apple's documented
`elements_per_row` / `matrix_origin` / `transpose_matrix` semantics so the
surrounding kernels are unchanged. The header defines macros that redirect the
kernel sources to the shim only when `DS4_METAL_SIMDGROUP_SHIM` is set.

### `ds4_metal.m`

* Adds `metal/simdgroup_shim.metal` as the first embedded Metal source.
* In `ds4_gpu_init`, detects `[g_device supportsFamily:MTLGPUFamilyApple7]`
  (the feature that gates the real builtins) and defines
  `DS4_METAL_SIMDGROUP_SHIM` on devices that lack it. Apple Silicon keeps the
  hardware path untouched. `DS4_METAL_DISABLE_SIMDGROUP_SHIM=1` forces it off.
* Makes residency-set creation non-fatal (warn and continue).

## Build and run

```sh
cd ds4
make            # Metal build: ./ds4 ./ds4-server ./ds4-bench ./ds4-eval ./ds4-agent
make cpu        # CPU-only build (overwrites the same names; default backend -> cpu)
./download_model.sh ds4f-q2
```

Metal (default on this host):

```sh
./ds4 -p "Say hello in one short sentence." -n 24 --temp 0
```

CPU:

```sh
./ds4 --cpu -p "Say hello in one short sentence." -n 16 --temp 0
```

## Measured results (greedy, same model and prompt)

| Backend | Prefill | Generation | Output |
|---|---:|---:|---|
| Metal, default GPU (W6800X) | 0.51–0.60 t/s | 0.15 t/s | coherent |
| CPU (`--cpu`), 28 threads | 3.09 t/s | 2.69 t/s | coherent |

GPU correctness check with the engine's own comparison
(`./ds4 --metal-graph-prompt-test -p "The capital of France is"`):

```
tokens=14 logits_max=6.0584 logits_rms=1.16618
cpu_top=19 gpu_top=19 cpu_top_logit=28.2007 gpu_top_logit=27.6117
```

CPU and GPU pick the **same** top token; the logits agree to ~2%. The portable
shim's own numeric tests (load/store round-trip with stride and transpose, an
8×8 half×half→float MAC, and `make_filled`) pass on the GPU.

## Why GPU generation is slower than CPU here

DwarfStar is designed for Apple Silicon unified memory: the whole model is
mapped once and every GPU kernel reads it at full system-RAM bandwidth. On a
discrete GPU there is no unified memory, so the shared model views live in host
RAM and the GPU reads them across PCIe every token. This model's 80.76 GiB is
almost entirely routed experts, so decode reads ~80 GiB/token over PCIe
(PCIe 3.0 ×16 ≈ 12 GB/s) → ~6.7 s/token, which is exactly the measured
0.15 t/s. The Xeon's system-RAM bandwidth makes the same reads ~18× faster on
the CPU.

The project's own reference points on unified-memory machines are 39 t/s
(M5 Max, 128 GB) and 18 t/s (DGX Spark) for this checkpoint; this host has a
different memory architecture, not a different code path.

## Worth doing next (not done)

* **Keep routed experts on the CPU and the dense/attention weights on the GPU**
  (the `-ncmoe` split llama.cpp uses on this machine; it reaches ~2.5–4.4 t/s).
  ds4 has no CPU-expert placement, so this is a real feature port, not a flag.
* **Multi-GPU placement** for Metal: the three cards hold 96 GiB of VRAM in
  total, enough for the whole checkpoint, but ds4's multi-device path is
  CUDA-only.
* An **expert cache in VRAM** (`--ssd-streaming` + `--ssd-streaming-cache-experts`)
  can cut PCIe traffic for hot experts but cannot remove it, so it is unlikely
  to close the gap to the CPU backend on its own.
