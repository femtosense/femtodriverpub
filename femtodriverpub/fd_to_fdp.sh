#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

########################################
# copy files from FD
FD_PATH=$HOME/femtodriver/femtodriver
FDP_PATH=$SCRIPT_DIR

MANIFEST=(
__init__.py
cfg.py
typing_help.py
addr_map_spu1p2.py
addr_map_spu1p3.py
program_handler.py
spu_runner.py
fx_runner.py
run/__init__.py
run/util.py
run/run_from_pt.py
plugins/io_plugin.py
plugins/__init__.py
plugins/zynq_plugin.py
plugins/fake_redis_client.py
plugins/redis_plugin.py
util/packing.py
util/hexfile.py
util/__init__.py
util/fake_socket.py
)

for file in ${MANIFEST[@]} ; do
    #echo "would copy $FD_PATH/$file to $FDP_PATH/$file"
    cp $FD_PATH/$file $FDP_PATH/$file
done

########################################
# mangle imports
FILES=`find . -name "*py"`
for fname in $FILES ; do 
    sed -i "s|femtodriver|femtodriverpub|g" $fname
done
