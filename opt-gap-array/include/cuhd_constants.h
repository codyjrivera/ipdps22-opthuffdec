/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_CONSTANTS_
#define CUHD_CONSTANTS_

// maximum codeword length this decoder can process
#define MAX_CODEWORD_LENGTH 14

// data type of a unit
#define UNIT_TYPE std::uint32_t

// data type of a symbol
#define SYMBOL_TYPE std::uint16_t

// data type for storing the bit length of codewords
#define BIT_COUNT_TYPE SYMBOL_TYPE // for aligned shared memory reads

#define DECODE_OUT_TYPE ushort2

#define DECODE_BUFFER_CAP 2

#endif /* CUHD_CONSTANTS_H_ */

