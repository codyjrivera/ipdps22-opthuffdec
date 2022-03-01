## Optimizing Huffman Decoding for Error-Bounded Lossy Compression on GPUs

by Cody Rivera [cjrivera1@crimson.ua.edu], Sheng Di [sdi1@crimson.ua.edu], 
Jiannan Tian [jiannan.tian@wsu.edu], Xiaodong Yu [xyu@anl.gov], Dingwen Tao [dingwen.tao@wsu.edu], and Franck Cappello [cappello@mcs.anl.gov]

This repository contains the four decoders we evaluated throughout our paper [1].

- orig-self-sync: Weissenberger and Schmidt, ICPP '18 [2, 3]
- orig-gap-array: Yamamoto et. al., ICPP '20 [4, 5]
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

# References

[1]. Cody Rivera, Sheng Di, Jiannan Tian, Xiaodong Yu, Dingwen Tao, and Franck Cappello, "Optimizing Huffman Decoding for Error-Bounded Lossy Compression on GPUs", To appear in *2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2022.

[2]. Andre Weissenberger and Bertil Schmidt. "Massively Parallel Huffman Decoding on GPUs", In *Proceedings of the 47th International Conference on Parallel Processing (ICPP)*, 2018, pp. 1-10.

[2]. Andre Weissenberger and Bertil Schmidt. "Massively Parallel Huffman Decoding on GPUs", In *Proceedings of the 47th International Conference on Parallel Processing (ICPP)*, 2018, pp. 1-10.

[3]. *CUHD: - A massively parallel Huffman decoder*. https://github.com/weissenberger/gpuhd

[4]. Naoya Yamamoto, Koji Nakano, Yasuaki Ito, Daisuke Takafuji, Akihiko Kasagi, and Tsuguchika Tabaru. “Huffman Coding with Gap Arrays for GPU Acceleration”. In *49th International
Conference on Parallel Processing (ICPP)*. 2020, pp. 1–11.

[5]. *Huffman_coding_Gap_arrays*, https://github.com/daisuke-takafuji/Huffman_coding_Gap_arrays


