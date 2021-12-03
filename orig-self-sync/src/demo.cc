/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <llhuff.h>     // encoder
#include <cuhd.h>       // decoder

// subsequence size
#define SUBSEQ_SIZE 4

// threads per block
#define NUM_THREADS 128

// performance parameter, high = high performance, low = low memory consumption
#define DEVICE_PREF 9

double p1Time[1024], p2Time[1024], p3Time[1024], p4Time[1024];

template <typename T>
T get_le_uint(char* buffer) {
    T result = 0;
    for (unsigned int i = 0; i < sizeof(T); ++i) {
        result += (uint8_t) buffer[i] << 8 * i;
    }
    return result;
}

int main(int argc, char** argv) {
    // name of the binary file
    const char* bin = argv[0];

    if(argc != 4) {
        std::cout << "USAGE: " << bin << " <compute device index> "
        << "<input file> <number of bytes>" << std::endl;
        return 1;
    }
    
    // compute device to use
    const std::int64_t compute_device_id = atoi(argv[1]);
    
    // input size in MB
    const long int size_bytes = atol(argv[3]);
    
    if(compute_device_id < 0 || size_bytes < 1) {
        std::cout << "USAGE: " << bin << " <compute device index> "
        << "<input file> <number of bytes>" << std::endl;
        return 1;
    }

    // vector for storing time measurements
    std::vector<std::pair<std::string, size_t>> timings;
    
    // load data from file
    const long int size = size_bytes / sizeof(SYMBOL_TYPE);
    std::vector<SYMBOL_TYPE> buffer;
    buffer.resize(size);

    std::cout << "File: " << argv[2] << std::endl;
   
    std::ifstream inf(argv[2], std::ifstream::binary);
    //uint16_t inb;
    char inb[sizeof(SYMBOL_TYPE)];
    TIMER_START(timings, "loading data");
    for (long int i = 0; i < size; ++i) {
        inf.read(inb, sizeof(SYMBOL_TYPE));
        if (!inf) {
            std::cerr << "Bad I/O" << std::endl;
            exit(1);
        }
        buffer[i] = get_le_uint<SYMBOL_TYPE>(inb); // little endian
    }
    TIMER_STOP;
    inf.close();

    std::shared_ptr<std::vector<llhuff::LLHuffmanEncoder::Symbol>> lengths;
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> enc_table;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table;
    
    // compress the random data using length-limited Huffman coding
    TIMER_START(timings, "generating coding tables")

        // determine optimal lengths for the codewords to be generated
        lengths = llhuff::LLHuffmanEncoder::get_symbol_lengths(
            buffer.data(), size);
    
        // generate encoder table
        enc_table = llhuff::LLHuffmanEncoder::get_encoder_table(lengths);
    
        // generate decoder table
        dec_table = llhuff::LLHuffmanEncoder::get_decoder_table(enc_table);
    TIMER_STOP
    
    // buffer for compressed data
    std::unique_ptr<UNIT_TYPE[]> compressed
        = std::make_unique<UNIT_TYPE[]>(enc_table->compressed_size);

    // compress
    TIMER_START(timings, "encoding")
        llhuff::LLHuffmanEncoder::encode_memory(compressed.get(),
            enc_table->compressed_size, buffer.data(), size, enc_table);
    TIMER_STOP
        
    // select CUDA device
	cudaSetDevice(compute_device_id);

    std::cout << "CR: "
              << (double) (size * sizeof(SYMBOL_TYPE)) / (double) (enc_table->compressed_size * sizeof(UNIT_TYPE))
              << std::endl;
    
	// define input and output buffers
    auto in_buf = std::make_shared<cuhd::CUHDInputBuffer>(
        reinterpret_cast<std::uint8_t*> (compressed.get()),
        enc_table->compressed_size * sizeof(UNIT_TYPE));
    
    auto out_buf = std::make_shared<cuhd::CUHDOutputBuffer>(size);

    auto gpu_in_buf = std::make_shared<cuhd::CUHDGPUInputBuffer>(in_buf);
    auto gpu_table = std::make_shared<cuhd::CUHDGPUCodetable>(dec_table);
    auto gpu_out_buf = std::make_shared<cuhd::CUHDGPUOutputBuffer>(out_buf);
    
    // auxiliary memory for decoding
    auto gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
        in_buf->get_compressed_size_units(),
        SUBSEQ_SIZE, NUM_THREADS);
    CUERR;
    
    // allocate gpu input and output buffers
    TIMER_START(timings, "GPU buffer allocation")
        gpu_in_buf->allocate();
        gpu_out_buf->allocate();
        gpu_table->allocate();
    TIMER_STOP

    // allocate auxiliary memory for decoding (sync info, etc.)
    TIMER_START(timings, "GPU/Host aux memory allocation")
        gpu_decoder_memory->allocate();
    TIMER_STOP
    
    // copy some data
    gpu_table->cpy_host_to_device();
    
    TIMER_START(timings, "GPU memcpy HtD")
        gpu_in_buf->cpy_host_to_device();
    TIMER_STOP
    
    // decode
        cuhd::CUHDGPUDecoder::decode(
            gpu_in_buf, in_buf->get_compressed_size_units(),
            gpu_out_buf, out_buf->get_uncompressed_size(),
            gpu_table, gpu_decoder_memory,
            MAX_CODEWORD_LENGTH, SUBSEQ_SIZE, NUM_THREADS);
    int constexpr NROUNDS = 10;                       
    TIMER_START(timings, "decoding")
        for (int i = 0; i < NROUNDS; ++i) {
        cuhd::CUHDGPUDecoder::decode(
            gpu_in_buf, in_buf->get_compressed_size_units(),
            gpu_out_buf, out_buf->get_uncompressed_size(),
            gpu_table, gpu_decoder_memory,
            MAX_CODEWORD_LENGTH, SUBSEQ_SIZE, NUM_THREADS);
        }
    TIMER_STOP
    
    // copy decoded data back to host
    TIMER_START(timings, "GPU memcpy DtH")
        gpu_out_buf->cpy_device_to_host();
    TIMER_STOP;

    // print timings
    for(auto &i: timings) {
        if (std::string("decoding") == i.first) {
            double timeUs = i.second / NROUNDS;
            std::cout << i.first << ".. "  << timeUs
                      << "µs" << std::endl;
            double thruGbs = ((double)(size * sizeof(SYMBOL_TYPE))
                              / (timeUs * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "throughput: " << thruGbs
                      << " GB/s" << std::endl;
        }
        else std::cout << i.first << ".. " << i.second << "µs" << std::endl;
    }

    double p1Avg = 0, p2Avg = 0, p3Avg = 0, p4Avg = 0;
    for (int i = 1; i <= NROUNDS; ++i) {
        p1Avg += p1Time[i]; p2Avg += p2Time[i];
        p3Avg += p3Time[i]; p4Avg += p4Time[i];
    }
    p1Avg /= NROUNDS; p2Avg /= NROUNDS;
    p3Avg /= NROUNDS; p4Avg /= NROUNDS;

    std::cout << "Detailed Profile (us):"
              << std::endl << "Phase 1: "
              << p1Avg
              << std::endl << "Phase 2: "
              << p2Avg
              << std::endl << "Phase 3: "
              << p3Avg
              << std::endl << "Phase 4: "
              << p4Avg
              << std::endl;
    
    // compare decompressed output to uncompressed input
    cuhd::CUHDUtil::equals(buffer.data(),
        out_buf->get_decompressed_data().get(), size) ? std::cout << std::endl
            : std::cout << std::endl << "mismatch" << std::endl;
    
    return 0;
}
