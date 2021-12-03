#!/bin/bash

CUSZ_DIR=
DATA_DIR=

dn=0

test2() {
    desc=$1
    errmode=$2
    file=$3
    size=$4

    echo "Testing $file: $desc (error mode $errmode)"
    
    for eb in 1e-3; do
        echo "Error bound $eb"
        $CUSZ_DIR/bin/cusz -t f32 --report time --config huffbyte=8 --skip huffman -m $errmode -e $eb -i $file -l $size -z --opath $DATA_DIR
        echo "Evaluating gap array code"
        ./centrify $file.quant $file.quant8 1024
        ./encoder/bin/encoder $file.quant8 compressedfile
        printf "CR: "
        echo "$(wc -c <$file.quant8) / $(wc -c <compressedfile)" | bc -l
        ./decoder/bin/decoder compressedfile
        diff $file.quant8 decodedfile
    done
}

test2 'HACC' r2r $DATA_DIR/vx.f32 280953867
test2 'EXAALT' r2r $DATA_DIR/dataset2-2338x106711.x.f32.dat 2338,106711
test2 'CESM Large' r2r $DATA_DIR/CLDICE_1_26_1800_3600.f32 26,1800,3600
test2 'nyx' r2r $DATA_DIR/baryon_density.dat 512,512,512
test2 'Isabel' r2r $DATA_DIR/HURR-CAT.bin.f32 400,500,500
