# Maya's GLM and DeepSeek GGUFs on Intel Arc (llama.cpp SYCL)

**Update, October 30, 2026.**

- **GLM-5.3-Flash Maya-S v2 runs on stock llama.cpp** ([v0.8.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.8.0),
  SYCL backend, no patch): see [GLM-5.3-Flash Maya-S v2](#glm-53-flash-maya-s-v2-stock-llamacpp) below.
- **DeepSeek:** llama.cpp#30817 was closed on October 27 in favour of
  [ggml-org/llama.cpp#31086](https://github.com/ggml-org/llama.cpp/pull/31086), now in releases. Its loader reads the
  layout of llama.cpp's own converter (`deepseek41.ple.*` keys, one shared `per_layer_token_embd`, and an
  `engram.token_map` Maya's files do not carry), so **Maya-XS still loads only on the #30817 head with the patch below**.
  Re-measured on October 29 (compute runtime 26.35.39758.17): every check passes, see [Results](#results).

## GLM-5.3-Flash Maya-S v2 (stock llama.cpp)

Maya's GLM GGUFs use the `glm5-next` architecture, which upstream llama.cpp supports. Nothing needs patching:

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout v0.8.0
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DNN=OFF
cmake --build build-sycl -j 8 --target llama-server
```

Maya-S v2 IQ2_XXS is 90 GB in three shards. On 88 GiB of VRAM the routed experts of six layers (about 11.4 GiB) stay
in host RAM and everything else goes to the GPUs:

```sh
M=GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ./build-sycl/bin/llama-server --model $M \
  --ctx-size 65536 --batch-size 1024 --ubatch-size 1024 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --fit off --jinja --threads 16 \
  --device SYCL0,SYCL1,SYCL2 --split-mode layer --n-gpu-layers all --tensor-split 17,16,14 \
  -ot 'blk\.(10|22|28|36|40|44)\.ffn_.*_exps\.=CPU' --no-host --no-op-offload --load-mode none
```

GLM has no Engram tables, so only the experts above live outside VRAM. Measured October 30, 2026 (same three cards,
compute runtime 26.35.39758.17, greedy, one stream, a 2,377-token prompt with 128 generated tokens):

| build | prefill tok/s | decode tok/s |
|---|---|---|
| llama.cpp v0.8.0, stock (second run; the first after load: 30.9 / 11.6) | 68.3 | 13.6 |
| llama.cpp v0.7.0 with a local grouped `MUL_MAT_ID` change for SYCL (not upstream), through a gateway | 108.9 | 14.0 |

Decode is the same; the grouped expert matmul only speeds up prompt reading. Answers are coherent; the model reasons
before it answers, so give it a budget above 128 tokens for real use.

## DeepSeek-V4.1-Flash-1017 Maya-XS (llama.cpp#30817 + loader patch)

Maya's engine is CUDA and HIP; it has no Intel backend. This page shows how to run Maya's own
**DeepSeek-V4.1-Flash-1017 Maya-XS** GGUFs on Intel Arc GPUs today, with llama.cpp's SYCL backend and the open
llama.cpp pull request that adds the `deepseek41` architecture
([ggml-org/llama.cpp#30817](https://github.com/ggml-org/llama.cpp/pull/30817)), plus one small loader patch so that
llama.cpp reads Maya's file layout. Nothing in Maya's engine or installer changes.

Measured on three Intel Arc Pro cards (B70 32 GB, B65 32 GB, B60 24 GB) in one Linux PC with a Ryzen 9 9950X and
64 GB of DDR5. Text only; Maya's DSpark drafter and the vision tower are not used on this path.

### Why a patch is needed

Maya's GGUFs and the converter in llama.cpp#30817 write the same model with two different layouts:

| | Maya GGUFs | llama.cpp#30817 converter |
|---|---|---|
| Engram metadata keys | `deepseek41.engram.{layers,max_ngram_size,head_count,head_dim,multipliers,offsets,primes}` | `deepseek41.ple.*`, `embedding_length_per_layer_input` |
| Engram offsets / primes | u32 | u64 |
| Engram tables | one per Engram layer (`blk.1.engram_embd`, `blk.14.engram_embd`), offsets local to each | one shared `per_layer_token_embd`, global offsets |
| `hash_layer_count` | absent (V4.1 has no hash-routed layers) | written as 0 |

Without the patch, llama.cpp stops with `key not found in model: deepseek41.hash_layer_count` and then
`wrong number of tensors; expected 1044, got 1038`. [patches/llama.cpp-deepseek41-maya-gguf.patch](patches/llama.cpp-deepseek41-maya-gguf.patch)
(against the head of #30817, `b72050d`) makes the loader accept both layouts: it reads either key set, widens u32
arrays, uses the per-layer tables when they are present (prefetch and host placement included), and makes
`hash_layer_count` optional for `deepseek41` only. Files from the #30817 converter load exactly as before.

### Build

oneAPI 2026.1 or 2026.2 and the Intel compute runtime (Level Zero) are assumed; `sycl-ls` lists the cards.

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch origin pull/30817/head:deepseek41 && git checkout deepseek41
git apply /path/to/project-maya/docs/patches/llama.cpp-deepseek41-maya-gguf.patch
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DEVICE_ARCH=bmg-g21,bmg-g31
cmake --build build-sycl -j 8 --target llama-server
```

The ahead-of-time device link of `ggml-sycl` can need a lot of host RAM; if it fails with `gen compiler command
failed with exit code 254`, lower `-fsycl-max-parallel-link-jobs` (6 worked with 64 GB of RAM).

### Run

Maya-XS is 276.6 GB: 107 GiB of weights (98.9 GiB routed experts, IQ1_S) in shards 1-3 and the two Engram tables
(150 GiB, Q6_K) in shards 4-5. With 88 GiB of VRAM across three cards, about a third of the expert layers stay in
host RAM and the Engram tables are read from the SSD by row.

```sh
M=DeepSeek-V4.1-Flash-1017-Maya-XS-IQ1_S-00001-of-00005.gguf
CPU_LAYERS='2|5|8|11|14|17|20|23|26|29|32|34|36|38'     # 14 expert layers in host RAM
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=0 ./build-sycl/bin/llama-server --model $M \
  --ctx-size 65536 --batch-size 1024 --ubatch-size 512 --parallel 1 --flash-attn on --fit off --jinja \
  --device SYCL0,SYCL1,SYCL2 --split-mode layer --n-gpu-layers all --tensor-split 14,15,12 \
  -ot "engram_embd=CPU,blk\.(${CPU_LAYERS})\.ffn_.*_exps\.=CPU" --no-host --load-mode mmap
```

Settings that matter, each learned the hard way:

- **`--load-mode mmap`, not `dio`.** With direct I/O the CPU-resident experts are copied into anonymous memory
  (36 GiB here), which ran a 64 GB host out of memory. With mmap they are page cache the kernel can reclaim.
- **Leave at least 4 GiB of VRAM free on every card.** `--tensor-split 15,15,10` over 41 layers put 31 of 32.6 GiB on
  the first card and left no room for compute buffers. `14,15,12` gave 27.9 / 28.1 / 17.8 GiB. KV (55 MiB at 8K) and
  compute buffers (about 150 MiB per card) are small.
- **Never `--fit`.** Its partial-layer overflow made the SYCL backend allocate driver memory outside any cgroup.
- **Give the server room for page cache.** In a systemd unit, `MemoryMax=40G` against `30G` raised decode on a
  1,934-token prompt from 9.97 to 10.92 tok/s and cut SSD reads during the benchmark from 34 to 28 GiB.

### Results

llama.cpp `llama-server`, greedy, one stream:

| prompt | prefill tok/s | decode tok/s |
|---|---|---|
| 17 tokens | 5.1 | 6.7 |
| 1,934 tokens | 45.6 | 10.9 |
| 260 tokens, cached prefix | 61.4 | 10.7 |
| 2,470 tokens, through an OpenAI-compatible gateway | 41.0 | 10.8 |
| 11,038 tokens, through the gateway (compute runtime 26.31) | 23.5 | 7.1 |
| 2,470 tokens, through the gateway, October 29 (runtime 26.35.39758.17; first request after a cold load) | 28.4 | 9.3 |

Checked: arithmetic (`17*23` = 391), an 11K-token prompt with exact recall of its first sentence, streaming,
reasoning content separated from the answer, and an OpenAI tool call with a tool-result round trip. The compute
runtime upgrade from 26.31 to 26.35 (IGC 2.41.5, Level Zero 1.34) raised decode on the 2,470-token prompt from 8.0 to
10.8 tok/s with the same binary.

`test-backend-ops` on the SYCL backend (B70 and B60): the ops of the `deepseek41` graph pass (LIGHTNING_INDEXER, TOP_K,
SOFTPLUS, GET_ROWS, SET_ROWS, MUL_MAT_ID, FLASH_ATTN_EXT at head size 512). MXFP8 `GET_ROWS` (the #30817 converter's
default Engram type) is not supported on SYCL, so use the Q6_K Engram tables in Maya's files.

## Known limits

- Text only. DSpark speculative drafting, tensor parallelism and Engram streaming are Maya-engine features and are not
  available on this path; this is llama.cpp's own placement.
- llama.cpp#30817 was closed unmerged; the patch applies to its head `b72050d`. Upstream's own DeepSeek-V4.1
  support (#31086) needs files in its converter's layout, which Maya-XS is not.
- On this machine the `xe` kernel driver occasionally reset a GPU's copy or compute engine under heavy multi-GPU
  load (kernel 7.0.0-38). Not root-caused; it happened with other models too.
- Measured on one machine only.
