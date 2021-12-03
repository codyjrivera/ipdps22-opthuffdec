## IPDPS '22 Paper 131 Code

This repository contains the four decoders we evaluated throughout our paper.

- orig-self-sync: Weissenberger and Schmidt, ICPP '18
- orig-gap-array: Yamamoto et. al., ICPP '20
- opt-self-sync: ICPP '18 plus optimizations
- opt-gap-array: ICPP '20 plus optimizations (and based on ICPP '18 codebase)

# Test platform information
- x86_64 CPU/Nvidia Tesla V100 GPU (32 GB)
- GCC 8.3.1
- NVCC 11.1

# Instructions
1. Download and compile cuSZ from https://github.com/szcompressor/cuSZ
2. Use the getdata.sh script as follows: `./getdata.sh dir` to download the datasets mentioned in the paper.
3. For the decoder you wish to examine, edit that code's `run.sh`, modifying the
   `CUSZ_DIR` to point to where your local cuSZ repository is, and modifying the `DATA_DIR`
   variable to point where you downloaded the data in Step 2.
4. Type `make` to build the code, and then type `./run.sh`.


(For `runcusz.sh`, which evaluates cuSZ's Huffman decoding, just follow step 3 and type `./runcusz.sh`)
(Note that our changes are mostly found in `demo.cc` and `cuhd_gpu_decoder.cu`)