/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_codetable.h"

cuhd::CUHDCodetable::CUHDCodetable(size_t cache_len, size_t num_entries)
    : cache_len_(cache_len),
      size_(1 << MAX_CODEWORD_LENGTH),
      num_entries_(num_entries),
      shared_table_(std::make_unique<CUHDCodetableItemSingle[]>(1 << cache_len)),
      table_(std::make_unique<CUHDCodetableItemSingle[]>(get_size())) {
    
}

size_t cuhd::CUHDCodetable::get_shared_size() {
    return 1 << cache_len_;
}

size_t cuhd::CUHDCodetable::get_cache_len() {
    return cache_len_;
}

size_t cuhd::CUHDCodetable::get_size() {
    return size_;
}

size_t cuhd::CUHDCodetable::get_num_entries() {
    return num_entries_;
}

size_t cuhd::CUHDCodetable::get_max_codeword_length() {
    return MAX_CODEWORD_LENGTH;
}

cuhd::CUHDCodetableItemSingle* cuhd::CUHDCodetable::get_shared() {
    return shared_table_.get();
}

cuhd::CUHDCodetableItemSingle* cuhd::CUHDCodetable::get() {
    return table_.get();
}

