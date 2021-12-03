/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_codetable.h"

cuhd::CUHDGPUCodetable::CUHDGPUCodetable(
    std::shared_ptr<CUHDCodetable> codetable)
    : shared_table_(codetable->get_shared(), codetable->get_shared_size()),
      table_(codetable->get(), codetable->get_size()) {
    
}

cuhd::CUHDCodetableItemSingle* cuhd::CUHDGPUCodetable::get_shared() {
    return shared_table_.get();
}

cuhd::CUHDCodetableItemSingle* cuhd::CUHDGPUCodetable::get() {
    return table_.get();
}

void cuhd::CUHDGPUCodetable::allocate() {
    shared_table_.allocate();
    table_.allocate();
}

void cuhd::CUHDGPUCodetable::free() {
    shared_table_.free();
    table_.free();
}

void cuhd::CUHDGPUCodetable::cpy_host_to_device() {
    shared_table_.cpy_host_to_device();
    table_.cpy_host_to_device();
}

void cuhd::CUHDGPUCodetable::cpy_device_to_host() {
    shared_table_.cpy_device_to_host();
    table_.cpy_device_to_host();
}
