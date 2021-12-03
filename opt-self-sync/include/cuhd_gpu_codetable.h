/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_CODETABLE_
#define CUHD_GPU_CODETABLE_

#include "cuhd_gpu_memory_buffer.h"
#include "cuhd_codetable.h"

namespace cuhd {
    class CUHDGPUCodetable {
    public:
        CUHDGPUCodetable(std::shared_ptr<CUHDCodetable> codetable);

        CUHDCodetableItemSingle* get_shared();
        CUHDCodetableItemSingle* get();
        
        void allocate();
        void free();
        void cpy_host_to_device();
        void cpy_device_to_host();

    private:
        CUHDGPUMemoryBuffer<CUHDCodetableItemSingle> shared_table_;
        CUHDGPUMemoryBuffer<CUHDCodetableItemSingle> table_;
    };
}

#endif /* CUHD_GPU_CODETABLE_H_ */

