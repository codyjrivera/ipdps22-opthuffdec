/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_decoder.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
//#include <curand_kernel.h>
#include "cub/cub.cuh"
#include <map>
#include "hist.cuh"

extern int shared_size;

extern double p1Time[1024], p2Time[1024], p3Time[1024], p4Time[1024], tuneTime[1024];

__device__ uint32_t* s1, *s2;

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) (((a)<(b))?(a):(b))

__global__ void globalize_stat_arrays(uint32_t* a, uint32_t* b) {
    s1 = a;
    s2 = b;
}

template <typename T>
__global__ void store_sequence_cr(T* out,
                                  std::uint32_t* indexOut,
                                  const T* in,
                                  std::uint32_t size,
                                  int clamp = -1) {
    std::uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size) {
        indexOut[gid] = gid;
        if (clamp >= 0) {
            out[gid] = clamp;
        } else {
            T value = in[gid];
            if (value <= 1 * 1024) {
                out[gid] = 0;
            } else if (value <= 2 * 1024) {
                out[gid] = 1;
            } else if (value <= 3 * 1024) {
                out[gid] = 2;
            } else if (value <= 4 * 1024) {
                out[gid] = 3;
            } else if (value <= 5 * 1024) {
                out[gid] = 4;
            } else if (value <= 6 * 1024) {
                out[gid] = 5;
            } else if (value <= 7 * 1024) {
                out[gid] = 6;
            } else if (value <= 8 * 1024) {
                out[gid] = 7;
            } else {
                out[gid] = 8;
            }
        }
    }
}

__device__ __forceinline__ void decode_subsequence(
    std::uint32_t subsequence_size,
    std::uint32_t current_subsequence,
    UNIT_TYPE shared_mask,
    UNIT_TYPE mask,
    std::uint32_t shared_shift,
    std::uint32_t shift,
    std::uint32_t start_bit,
    std::uint32_t &in_pos,
    const UNIT_TYPE* __restrict__ in_ptr,
    UNIT_TYPE subsequence[4],
    UNIT_TYPE &overflow_unit,
    //UNIT_TYPE &window,
    //UNIT_TYPE &next,
    std::uint32_t &last_word_unit,
    std::uint32_t &last_word_bit,
    std::uint32_t &num_symbols,
    std::uint32_t &out_pos,
    std::uint16_t* out_ptr,
    std::uint32_t &next_out_pos,
    const cuhd::CUHDCodetableItemSingle* __restrict__ shared_table,
    const int cache_len,
    const cuhd::CUHDCodetableItemSingle* __restrict__ table,
    const std::uint32_t bits_in_unit,
    std::uint32_t &last_at,
    bool overflow,
    bool write_output) {

    // local unit registers
    UNIT_TYPE work_window
        = (overflow && current_subsequence > 0) ? overflow_unit : subsequence[0];
    UNIT_TYPE work_next
        = (overflow && current_subsequence > 0) ? subsequence[0] : subsequence[1];
    
    // Output buffering
//    DECODE_OUT_TYPE out_buffer; // of size DECODE_BUFFER_CAP
    //  std::uint32_t out_buffer_count = 0;
    //std::uint32_t out_buffer_pos = 0;

    // current unit in this subsequence
    std::uint32_t current_unit = 0;
    
    // current bit position in unit
    std::uint32_t at = start_bit;

    // number of symbols found in this subsequence
    std::uint32_t num_symbols_l = 0;

    // shift to start
    UNIT_TYPE copy_next = work_next;
    if (at > 0) {
        copy_next >>= bits_in_unit - at;

        work_next <<= at;
        work_window <<= at;
        work_window += copy_next;
    }

    // perform overflow from previous subsequence
    if(overflow && current_subsequence > 0) {

        // decode first symbol
        std::uint32_t taken = shared_table[(work_window & shared_mask) >> shared_shift].num_bits;
        if (taken == 0)
            taken = table[(work_window & mask) >> shift].num_bits;

        copy_next = work_next;
        copy_next >>= bits_in_unit - taken;

        work_next <<= taken;
        work_window <<= taken;
        at += taken;
        work_window += copy_next;

        // overflow
        if(at > bits_in_unit) {
            ++in_pos;
            work_window = subsequence[0];
            work_next = subsequence[1];
            at -= bits_in_unit;
            work_window <<= at;
            work_next <<= at;

            copy_next = subsequence[1]; //in_ptr[in_pos + 1];
            copy_next >>= bits_in_unit - at;
            work_window += copy_next;
        }

        else {
            ++in_pos;
            work_window = subsequence[0];
            work_next = subsequence[1];
            at = 0;
        }
    }
    
    while(current_unit < subsequence_size) {
        
        while(at < bits_in_unit) {
            cuhd::CUHDCodetableItemSingle hit =
                shared_table[(work_window & shared_mask) >> shared_shift];
            
            if (hit.num_bits == 0) {
                hit = table[(work_window & mask) >> shift];
            }

            //if (15 <= out_pos && out_pos <= 25) {
            //printf("At (gid %d) bit %d in unit %d (true unit %d), decoded %d to %d\n", threadIdx.x + blockIdx.x * blockDim.x, at, current_unit, in_pos, hit.symbol, out_pos);         
            //}
            
            // decode a symbol
            std::uint32_t taken = hit.num_bits;
            ++num_symbols_l;

            if(write_output) {
                // if (out_pos + DECODE_BUFFER_CAP <= next_out_pos && out_pos % DECODE_BUFFER_CAP == 0) {
//                     out_buffer_pos = out_pos / DECODE_BUFFER_CAP;
//                     out_buffer.x = hit.symbol;
//                     out_buffer_count = 1;
//                     ++out_pos;
//                 } else if (out_buffer_count > 0 && out_buffer_count < DECODE_BUFFER_CAP) {
//                     // Perhaps replace with constexpr if or something else nicer
//                     switch (out_buffer_count) {
// //#if DECODE_BUFFER_CAP <= 2
//                     case 1: 
//                         out_buffer.y = hit.symbol;
//                         break;
//                     } // TODO add for larger vectors
//                     ++out_buffer_count;
//                     ++out_pos;
//                     if (out_buffer_count == DECODE_BUFFER_CAP) {
//                         ((DECODE_OUT_TYPE*)out_ptr)[out_buffer_pos] = out_buffer; 
//                         out_buffer_count = 0;
//                     }
//                 } else if (out_pos < next_out_pos) {
//                     out_ptr[out_pos] = hit.symbol;
//                     ++out_pos;
//                 }
                if (out_pos < next_out_pos) {
                    out_ptr[out_pos] = hit.symbol;
                    ++out_pos;
                }
            }
            
            UNIT_TYPE copy_next = work_next;
            copy_next >>= bits_in_unit - taken;

            work_next <<= taken;
            work_window <<= taken;
            last_word_bit = at;
            at += taken;
            work_window += copy_next;
            last_word_unit = current_unit;
        }
        
        // refill decoder window if necessary
        ++current_unit;
        // Bugfix -- Don't refill window at last codeword
        if (current_unit < subsequence_size) {
            ++in_pos;

            if (current_unit < subsequence_size - 1) {
                // Make sure subsequence[] is allocated as registers
                #pragma unroll
                for (int i = 1; i <= 2; ++i) {
                    if (current_unit == i) {
                        work_window = subsequence[i];
                        work_next = subsequence[i + 1];
                    }
                }
            } else {
                work_window = overflow_unit = subsequence[3];
                *((uint4*) subsequence) = ((uint4*) in_ptr)[(in_pos + 1) / 4];
                work_next = subsequence[0];
            }
            
            if(at == bits_in_unit) {
                at = 0;
            } else {
                at -= bits_in_unit;
                work_window <<= at;
                work_next <<= at;
                
                UNIT_TYPE copy_next;
                if (current_unit < subsequence_size - 1) {
                    #pragma unroll
                    for (int i = 1; i <= 2; ++i) {
                        if (current_unit == i) {
                            copy_next = subsequence[i + 1];
                        }
                    }
                } else {
                    copy_next = subsequence[0];
                }
                copy_next >>= bits_in_unit - at;
                work_window += copy_next;
            }
        }
    }
    
    num_symbols = num_symbols_l;
    last_at = last_word_bit;
}

__global__ void phase1_decode_subseq(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    const UNIT_TYPE* __restrict__ in_ptr,
    const uint8_t* __restrict__ gap_array_ptr,
    cuhd::CUHDCodetableItemSingle* global_shared_table,
    const int cache_len,
    cuhd::CUHDCodetableItemSingle* table,
    uint4* sync_points,
    const std::uint32_t bits_in_unit) {

    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(gid < total_num_subsequences) {

        std::uint32_t current_subsequence = gid;
        //std::uint32_t current_subsequence_in_block = threadIdx.x;
        std::uint32_t in_pos = gid * subsequence_size;

        // mask
        const UNIT_TYPE shared_mask = ~(((UNIT_TYPE) (0) - 1) >> cache_len);
        const UNIT_TYPE mask = ~(((UNIT_TYPE) (0) - 1) >> table_size);

        // shift right
        const std::uint32_t shared_shift = bits_in_unit - cache_len;
        const std::uint32_t shift = bits_in_unit - table_size;

        std::uint32_t out_pos = 0;
        std::uint32_t next_out_pos = 0;
        std::uint16_t* out_ptr = 0;

        // sliding window
        //UNIT_TYPE window = in_ptr[in_pos];
        //UNIT_TYPE next = in_ptr[in_pos + 1];
        UNIT_TYPE subsequence[4];
        UNIT_TYPE overflow_unit;
        *((uint4*) subsequence) = ((uint4*) in_ptr)[in_pos / 4];

        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;

        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
        
        // bit position of next codeword -- Incorporate gap array shift
        std::uint32_t last_at = gap_array_ptr[gid];
        
        decode_subsequence(subsequence_size, current_subsequence,
                           shared_mask, mask, shared_shift, shift, last_at, in_pos, in_ptr, subsequence, overflow_unit,
                           last_word_unit, last_word_bit, num_symbols,
                           out_pos, out_ptr, next_out_pos, global_shared_table, cache_len,
                           table, bits_in_unit, last_at, false, false);
        
        sync_points[current_subsequence] =
            {last_word_unit, last_word_bit, num_symbols, 1};
    }    
}

__global__ void phase2_synchronise_blocks(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    std::uint32_t num_blocks,
    const UNIT_TYPE* __restrict__ in_ptr,
    cuhd::CUHDCodetableItemSingle* global_shared_table,
    const int cache_len,
    cuhd::CUHDCodetableItemSingle* table,
    uint4* sync_points,
    SYMBOL_TYPE* block_synchronized,
    const std::uint32_t bits_in_unit) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t num_of_seams = num_blocks - 1;

    if(gid < num_of_seams) {
    
        // mask
        const UNIT_TYPE shared_mask = ~(((UNIT_TYPE) (0) - 1) >> cache_len);
        const UNIT_TYPE mask = ~(((UNIT_TYPE) (0) - 1) >> table_size);

        // shift right
        const std::uint32_t shared_shift = bits_in_unit - cache_len;
        const std::uint32_t shift = bits_in_unit - table_size;

        std::uint32_t out_pos = 0;
        std::uint32_t next_out_pos = 0;
        std::uint16_t* out_ptr = 0;
        
        // jump to first sequence of the block
        std::uint32_t current_subsequence = (gid + 1) * blockDim.x;
        
        // search for synchronized sequences at the end of previous block
        uint4 sync_point = sync_points[current_subsequence - 1];
        
        // current unit
        std::uint32_t in_pos = (current_subsequence - 1) * subsequence_size;
        std::uint32_t old_in_pos = in_pos;
        
        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;

        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
       
        std::uint32_t last_at = sync_point.y;
        
        in_pos += sync_point.x;
        
        // sliding window
        //UNIT_TYPE window = in_ptr[in_pos];
        //UNIT_TYPE next = in_ptr[in_pos + 1];
        UNIT_TYPE subsequence[4];
        UNIT_TYPE overflow_unit = in_ptr[in_pos];
        *((uint4*) subsequence) = ((uint4*) in_ptr)[(old_in_pos / 4) + 1];
        
        std::uint32_t subsequences_processed = 0;
        bool synchronized_flag = false;

        while(subsequences_processed < blockDim.x) {
        
            if(!synchronized_flag) {
                decode_subsequence(subsequence_size, current_subsequence,
                    shared_mask, mask, shared_shift, shift, last_at, in_pos, in_ptr,
                    subsequence, overflow_unit, last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, global_shared_table, cache_len, 
                    table, bits_in_unit, last_at, true, false);
                
                sync_point = sync_points[current_subsequence];
                
                // if sync point detected
                if(sync_point.x == last_word_unit
                    && sync_point.y == last_word_bit) {
                        sync_point.z = num_symbols;
                        sync_point.w = 1;
                    
                        block_synchronized[gid + 1] = 1;
                        synchronized_flag = true;
                }
                
                // correct erroneous position data
                else {
                    sync_point.x = last_word_unit;
                    sync_point.y = last_word_bit;
                    sync_point.z = num_symbols;
                    sync_point.w = 0;
                    block_synchronized[gid + 1] = 0;
                }
                
                sync_points[current_subsequence] = sync_point;
            }

            ++current_subsequence;
            ++subsequences_processed;
            
            __syncthreads();
        }
    }
}

__global__ void phase3_copy_num_symbols_from_sync_points_to_aux(
    std::uint32_t total_num_subsequences,
    const uint4* __restrict__ sync_points,
    std::uint32_t* subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        subsequence_output_sizes[gid] = sync_points[gid].z;
    }
}

__global__ void phase3_copy_num_symbols_from_aux_to_sync_points(
    std::uint32_t total_num_subsequences,
    uint4* sync_points,
    std::uint32_t* sequence_sizes,
    const std::uint32_t* __restrict__ subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t total_num_sequences = gridDim.x;

    if (blockIdx.x <= total_num_sequences) {
        sequence_sizes[blockIdx.x]
            = subsequence_output_sizes[MIN(blockDim.x * (blockIdx.x + 1), total_num_subsequences - 1)]
            - subsequence_output_sizes[blockDim.x * blockIdx.x];
    }
    
    if(gid < total_num_subsequences) {
        sync_points[gid].z = subsequence_output_sizes[gid];
    }
}

__global__ void phase4_decode_write_output(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    const UNIT_TYPE* __restrict__ in_ptr,
    SYMBOL_TYPE* out_ptr,
    std::uint32_t output_size,
    cuhd::CUHDCodetableItemSingle* global_shared_table,
    const int cache_len,
    cuhd::CUHDCodetableItemSingle* table,
    const uint4* __restrict__ sync_points,
    const std::uint32_t bits_in_unit,
    const std::uint32_t shared_size,
    std::uint32_t* sequence_indices,
    std::uint32_t start_index) {

    extern __shared__ std::uint16_t shared_ptr[];
    __shared__ std::uint32_t subsequence_base, subsequence_offset, symbol_base, symbol_offset,
        write_start_pos, write_end_pos;

    std::uint32_t sequence_index = sequence_indices[blockIdx.x + start_index];
    
    std::uint32_t gid = blockDim.x * sequence_index + threadIdx.x;

    std::uint32_t block_start_pos = sync_points[blockDim.x * sequence_index].z;
    std::uint32_t block_end_pos = (blockDim.x * (sequence_index + 1) < total_num_subsequences) ?
        sync_points[blockDim.x * (sequence_index + 1)].z : output_size;

    if (/*block_end_pos - block_start_pos >= 2048*/0) {
        s1[gid] = 0;
        return;
    }
    
    // bool block_start_odd = block_start_pos % 2 == 1;
    // bool block_end_odd = (block_end_pos - 1) % 2 == 1;
    
    // std::uint32_t block_base_aligned = block_start_pos - block_start_odd;
    // std::uint32_t block_start_aligned = block_start_pos + block_start_odd;
    // std::uint32_t block_end_aligned = block_end_pos - block_end_odd;     

    subsequence_base = 0;
    subsequence_offset = blockDim.x;
    symbol_base = 0;
    symbol_offset = 0;
    write_start_pos = block_start_pos;
    write_end_pos = block_end_pos;
    __syncthreads();
    
    while (subsequence_base < blockDim.x) {
        if (gid < total_num_subsequences) {
            // mask
            const UNIT_TYPE shared_mask = ~(((UNIT_TYPE) (0) - 1) >> cache_len);
            const UNIT_TYPE mask = ~(((UNIT_TYPE) (0) - 1) >> table_size);

            // shift right
            const std::uint32_t shared_shift = bits_in_unit - cache_len;
            const std::uint32_t shift = bits_in_unit - table_size;

            // start bit of last codeword in this subsequence
            std::uint32_t last_word_unit = 0;
            std::uint32_t last_word_bit = 0;
        
            // number of symbols found in this subsequence
            std::uint32_t num_symbols = 0;
        
            // bit position of next codeword
            std::uint32_t last_at = 0;
        
            std::uint32_t current_subsequence = gid;
            std::uint32_t in_pos = current_subsequence * subsequence_size;
        
            uint4 sync_point = sync_points[current_subsequence];
            uint4 next_sync_point;
            if (gid == total_num_subsequences - 1) {
                next_sync_point.x = 0; next_sync_point.y = 0;
                next_sync_point.z = 0; next_sync_point.w = 0;
            } else {
                next_sync_point = sync_points[current_subsequence + 1];
            }
        
            std::uint32_t out_pos = sync_point.z - block_start_pos - symbol_base;
            std::uint32_t next_out_pos
                = (gid == total_num_subsequences - 1 ? output_size : next_sync_point.z) 
                - block_start_pos - symbol_base;

            if (out_pos < shared_size && next_out_pos >= shared_size) {
                subsequence_offset = threadIdx.x;
                symbol_offset = out_pos;
                write_end_pos = write_start_pos + out_pos;
            } else if (out_pos < shared_size /*&& next_out_pos < shared_size*/) {
                
                if (gid > 0) {
                    sync_point = sync_points[current_subsequence - 1];
                    in_pos = (current_subsequence - 1) * subsequence_size;
                }
        
                // sliding window
                //UNIT_TYPE window = in_ptr[in_pos];
                //UNIT_TYPE next = in_ptr[in_pos + 1];
                UNIT_TYPE subsequence[4];
                UNIT_TYPE overflow_unit;
                //*((uint4*) subsequence) = ((uint4*) in_ptr)[in_pos / 4];
        
                // start bit
                std::uint32_t start = 0;
        
                if(gid > 0) {
                    int old_in_pos = in_pos;
            
                    in_pos += sync_point.x;
                    start = sync_point.y;

                    overflow_unit = in_ptr[in_pos];
                    *((uint4*) subsequence) = ((uint4*) in_ptr)[(old_in_pos / 4) + 1];
                } else {
                    *((uint4*) subsequence) = ((uint4*) in_ptr)[in_pos / 4];
                }

                s1[gid] = (next_out_pos - out_pos > 0) ? next_out_pos - out_pos : 0;
                
                // overflow from previous subsequence, decode, write output
                decode_subsequence(subsequence_size, current_subsequence, shared_mask, mask, shared_shift, shift,
                                   start, in_pos, in_ptr, subsequence, overflow_unit,
                                   last_word_unit, last_word_bit, num_symbols, out_pos, shared_ptr,
                                   next_out_pos, global_shared_table, cache_len, table,
                                   bits_in_unit, last_at, true, true);
            }
        }
        __syncthreads();
        for (std::uint32_t i = write_start_pos + threadIdx.x;
             i < write_end_pos; i += blockDim.x) {
            out_ptr[i] = shared_ptr[i - write_start_pos];
        }
        gid += subsequence_offset;
        if (threadIdx.x == 0) {
            subsequence_base += subsequence_offset;
            symbol_base += symbol_offset;
            write_start_pos += symbol_offset;
            write_end_pos = block_end_pos;
            subsequence_offset = blockDim.x;
            symbol_offset = 0;
        }
        __syncthreads();
    }
}

int invocation_count = 0;

struct smallest_greater_percent {
    double percent;
    uint32_t maximum;
    
    __host__ __device__
    smallest_greater_percent(double p, uint32_t m) : percent(p), maximum(m) {}
    __host__ __device__
    bool operator()(uint32_t lhs, uint32_t rhs) {
        if (((double) lhs / maximum) < percent) {
            return false;
        }
        if (((double) rhs / maximum) < percent) {
            return true;
        }
        return lhs < rhs;
    }
};

int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

void cuhd::CUHDGPUDecoder::decode(
    std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
    size_t input_size,
    std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
    size_t output_size,
    std::shared_ptr<cuhd::CUHDGPUMemoryBuffer<uint8_t>> gap_array,
    size_t gap_array_size,
    std::shared_ptr<cuhd::CUHDGPUCodetable> table,
    int cache_len,
    std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
    size_t max_codeword_length,
    size_t preferred_subsequence_size,
    size_t threads_per_block) {
    
    ++invocation_count;
    
    UNIT_TYPE* in_ptr = input->get();
    SYMBOL_TYPE* out_ptr = output->get();
    uint8_t* gap_array_ptr = gap_array->get();
    cuhd::CUHDCodetableItemSingle* table_shared_ptr = table->get_shared();
    cuhd::CUHDCodetableItemSingle* table_ptr = table->get();

    // Shared memory configuration
    size_t shared_table_size = (1 << cache_len);
    
    uint4* sync_info = reinterpret_cast<uint4*>(aux->get_sync_info());
    std::uint32_t* output_sizes = aux->get_output_sizes();
    SYMBOL_TYPE* sequence_synced_device = aux->get_sequence_synced_device();
    SYMBOL_TYPE* sequence_synced_host = aux->get_sequence_synced_host();

    size_t num_subseq = SDIV(input_size, preferred_subsequence_size);
    size_t num_sequences = SDIV(num_subseq, threads_per_block);

    uint32_t *stat1, *stat2;
    uint32_t *dev_stat1, *dev_stat2;

    stat1 = new uint32_t[num_subseq];
    stat2 = new uint32_t[num_subseq];

    cudaMalloc(&dev_stat1, num_subseq * sizeof(uint32_t));
    cudaMalloc(&dev_stat2, num_subseq * sizeof(uint32_t));
    CUERR;

    // Optimization
    const uint32_t NUM_STREAMS = 9; // 8 compression ratios + one "overflow" stream
    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }


    uint32_t *dev_sequence_count;
    uint32_t *dev_sequence_cr, *dev_sequence_cr2;
    uint32_t *dev_sequence_index, *dev_sequence_index2;
    uint32_t *dev_histogram;
    uint32_t host_histogram[512], host_start_index[512];
    void *dev_sort_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaMalloc(&dev_sequence_count, num_sequences * sizeof(uint32_t));
    cudaMalloc(&dev_sequence_cr, num_sequences * sizeof(uint32_t));
    cudaMalloc(&dev_sequence_cr2, num_sequences * sizeof(uint32_t));
    cudaMalloc(&dev_sequence_index, num_sequences * sizeof(uint32_t));
    cudaMalloc(&dev_sequence_index2, num_sequences * sizeof(uint32_t));
    cudaMalloc(&dev_histogram, 512 * sizeof(uint32_t)); // (this should be algorithm-dependent).
    cub::DeviceRadixSort::SortPairs(dev_sort_storage, temp_storage_bytes,
                                    dev_sequence_cr, dev_sequence_cr2,
                                    dev_sequence_index, dev_sequence_index2,
                                    num_sequences);
    cudaMalloc(&dev_sort_storage, temp_storage_bytes);
    cudaMemset(dev_histogram, 0, 512 * sizeof(uint32_t));
    CUERR;
    
    globalize_stat_arrays<<<1,1>>>(dev_stat1, dev_stat2);
    
    const std::uint32_t bits_in_unit = sizeof(UNIT_TYPE) * 8;

    auto p1Before = std::chrono::steady_clock::now();
    // launch phase 1 (intra-sequence synchronisation)
    phase1_decode_subseq<<<num_sequences, threads_per_block, shared_table_size * sizeof(CUHDCodetableItemSingle)>>>(
        preferred_subsequence_size,
        num_subseq,
        max_codeword_length,
        in_ptr,
        gap_array_ptr,
        table_shared_ptr,
        cache_len,
        table_ptr,
        sync_info,
        bits_in_unit);
    cudaDeviceSynchronize();
    CUERR
    auto p1After = std::chrono::steady_clock::now();

    if (invocation_count == 0) {
        cudaMemcpy(stat1, dev_stat1, num_subseq * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < num_subseq; ++i) {
            std::cout << "GM1: " << i << "," << stat1[i] << std::endl;
        }
        CUERR;
    }
   
    //goto results;
    
    // launch phase 2 (inter-sequence synchronisation)
    auto p2Before = std::chrono::steady_clock::now();
    auto p2After = std::chrono::steady_clock::now();
    
    // launch phase 3 (parallel prefix sum)
    auto p3Before = std::chrono::steady_clock::now();
    thrust::device_ptr<std::uint32_t> thrust_sync_points(output_sizes);

    phase3_copy_num_symbols_from_sync_points_to_aux<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, output_sizes);
    CUERR

    thrust::exclusive_scan(thrust_sync_points,
        thrust_sync_points + num_subseq, thrust_sync_points);

    phase3_copy_num_symbols_from_aux_to_sync_points<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, dev_sequence_count, output_sizes);
    cudaDeviceSynchronize();
    CUERR
    auto p3After = std::chrono::steady_clock::now();
    
    auto tuneBefore = std::chrono::steady_clock::now();
    thrust::device_ptr<std::uint32_t> thrust_histogram(dev_histogram);
    thrust::device_ptr<std::uint32_t> thrust_sequence_index(dev_sequence_index);
    thrust::device_ptr<std::uint32_t> thrust_sequence_cr(dev_sequence_cr);

    int clamp = (shared_size == -1) ? -1 : 0;

    store_sequence_cr<<<SDIV(num_sequences, threads_per_block), threads_per_block>>>(dev_sequence_cr, dev_sequence_index,
                                                                                     dev_sequence_count, num_sequences, clamp);
    cudaDeviceSynchronize();
    CUERR;

    float dummy;
    wrapper::get_frequency(dev_sequence_cr, num_sequences, dev_histogram, 32, dummy);

    cub::DeviceRadixSort::SortPairs(dev_sort_storage, temp_storage_bytes,
                                    dev_sequence_cr, dev_sequence_cr2,
                                    dev_sequence_index, dev_sequence_index2,
                                    num_sequences);

    std::uint32_t* temp;
    temp = dev_sequence_cr;
    dev_sequence_cr = dev_sequence_cr2;
    dev_sequence_cr2 = temp;
    temp = dev_sequence_index;
    dev_sequence_index = dev_sequence_index2;
    dev_sequence_index2 = temp;

    cudaMemcpy(host_histogram, dev_histogram, 512 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    CUERR;
    host_start_index[0] = 0;
    for (int i = 1; i < NUM_STREAMS; ++i) {
        host_start_index[i] = host_start_index[i - 1] + host_histogram[i - 1];
    }
    
    auto tuneAfter = std::chrono::steady_clock::now();
    
    auto p4Before = std::chrono::steady_clock::now();
    // launch phase 4 (final decoding)
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int shmem_size;
        if (shared_size != -1) {
            shmem_size = shared_size;
        } else {
            if (i == NUM_STREAMS - 1) {
                shmem_size = 3584;
            } else {
                shmem_size = (i + 1) * 1024;
            }
        }

        if (host_histogram[i] > 0) {
            //std::cout << "Starting " << i << " with " << host_histogram[i] << " from " << host_start_index[i] << " to " <\
< host_start_index[i] + host_histogram[i] << std::endl;
            phase4_decode_write_output<<<host_histogram[i], threads_per_block, shmem_size * sizeof(uint16_t), streams[i]>>> \
                (
                    preferred_subsequence_size,
                    num_subseq,
                    max_codeword_length,
                    in_ptr,
                    out_ptr,
                    output_size,
                    table_shared_ptr,
                    cache_len,
                    table_ptr,
                    sync_info,
                    bits_in_unit,
                    shmem_size,
                    dev_sequence_index,
                    host_start_index[i]);
        }
    }
    cudaDeviceSynchronize();
    CUERR
    auto p4After = std::chrono::steady_clock::now();

    if (false) {
        cudaMemcpy(stat1, dev_stat1, num_subseq * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::map<uint32_t, uint32_t> thread_histogram;
        for (size_t i = 0; i < num_subseq; ++i) {
            ++thread_histogram[stat1[i]];
        }
        //std::cout << "total elts " << num_subseq << " total blocks " << num_subseq / 128 << std::endl;
        //for (auto e : thread_histogram) {
        //    std::cout << e.first << " elements " << e.second << " times" << std::endl;
        //}
        //std::cout << "Block histogram" << std::endl;
        std::map<uint32_t, uint32_t> block_histogram;
        std::cout << num_subseq << " " << num_subseq / 128 << std::endl;
        for (size_t i = 0; i < num_subseq / 128; ++i) {
            uint32_t total_block = 0;
            for (size_t j = 0; j < 128; ++j) {
                total_block += stat1[j + 128 * i];
            }
            ++block_histogram[total_block];
        }
        {
            size_t count = 0;
            bool pflag[4] = {false, false, false, false};
            std::cout << "num_subseq / 128: " << num_subseq << std::endl;
            for (auto e : block_histogram) {
                count += e.second;
                double progress = (double) count / (double) (num_subseq / 128);
                if (0.55 <= progress && !pflag[0]) {
                    std::cout << "60%: " << e.first << std::endl;
                    pflag[0] = true;
                }
                if (0.65 <= progress && !pflag[1]) {
                    std::cout << "70%: " << e.first << std::endl;
                    pflag[1] = true;
                }
                if (0.75 <= progress && !pflag[2]) {
                    std::cout << "80%: " << e.first << std::endl;
                    pflag[2] = true;
                }
                if (0.85 <= progress && !pflag[3]) {
                    std::cout << "90%: " << e.first << std::endl;
                    pflag[3] = true;
                }
                //std::cout << e.first << " elements " << e.second << " times" << std::endl;
            }
            std::cout << "max of count: " << count << std::endl;
        }
        CUERR;
    }
    
    p1Time[invocation_count - 1] = std::chrono::duration_cast<std::chrono::microseconds>(p1After - p1Before).count();
    p2Time[invocation_count - 1] = std::chrono::duration_cast<std::chrono::microseconds>(p2After - p2Before).count();
    p3Time[invocation_count - 1] = std::chrono::duration_cast<std::chrono::microseconds>(p3After - p3Before).count();
    p4Time[invocation_count - 1] = std::chrono::duration_cast<std::chrono::microseconds>(p4After - p4Before).count();
    tuneTime[invocation_count - 1] = std::chrono::duration_cast<std::chrono::microseconds>(tuneAfter - tuneBefore).count();

    delete[] stat1;
    delete[] stat2;
    cudaFree(dev_stat1);
    cudaFree(dev_stat2);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(dev_sequence_count);
    cudaFree(dev_sequence_cr);
    cudaFree(dev_sequence_index);
    cudaFree(dev_sequence_cr2);
    cudaFree(dev_sequence_index2);
    cudaFree(dev_histogram);
    cudaFree(dev_sort_storage);
    CUERR;
}

 