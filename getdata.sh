
if [ "$#" -ne 1 ]; then
    echo "usage: $0 data-dir";
    exit 2;
fi

if ! mkdir -p $1; then
    echo "Cannot download data to $1";
    exit 2;
fi

DATA_DIR=$1

echo "Downloading Data to $DATA_DIR";

echo "HACC 1GB";
if [ ! -f "$DATA_DIR/EXASKY-HACC-data-medium-size.tar.gz" ]; then
    wget -c -P $DATA_DIR https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-medium-size.tar.gz
fi

if [ ! -d "$DATA_DIR/280953867" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/EXASKY-HACC-data-medium-size.tar.gz
fi

cp $DATA_DIR/280953867/vx.f32 $DATA_DIR

echo "EXAALT";
if [ ! -f "$DATA_DIR/SDRBENCH-exaalt-helium.tar.gz" ]; then
    wget -c -P $DATA_DIR https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXAALT/SDRBENCH-exaalt-helium.tar.gz
fi

if [ ! -d "$DATA_DIR/SDRBENCH-exaalt-helium" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-exaalt-helium.tar.gz
fi

cp $DATA_DIR/SDRBENCH-exaalt-helium/dataset2-2338x106711.x.f32.dat $DATA_DIR

echo "CESM";
if [ ! -f "$DATA_DIR/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz" ]; then
    wget -c -P $DATA_DIR https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/CESM-ATM/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz
fi

if [ ! -d "$DATA_DIR/SDRBENCH-CESM-ATM-26x1800x3600" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz
fi

cp $DATA_DIR/SDRBENCH-CESM-ATM-26x1800x3600/CLDICE_1_26_1800_3600.f32 $DATA_DIR

echo "Nyx";
if [ ! -f "$DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz" ]; then
    wget -c -P $DATA_DIR https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
fi

if [ ! -d "$DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
fi

cp $DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.dat $DATA_DIR

echo "Hurricane Isabel";
if [ ! -f "$DATA_DIR/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz" ]; then
    wget -c -P $DATA_DIR https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
fi

if [ ! -d "$DATA_DIR/100x500x500" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
fi

PRE=$DATA_DIR/100x500x500/
cat $PRE/CLOUDf48.bin.f32 $PRE/PRECIPf48.bin.f32 $PRE/QCLOUDf48.bin.f32 $PRE/QRAINf48.bin.f32  >$DATA_DIR/HURR-CAT.bin.f32



