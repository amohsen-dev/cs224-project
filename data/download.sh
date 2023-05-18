#!/bin/bash
for i in $(seq $1 $2)
do
        wget -O "expert_data/games$i.lin" "http://www.bridgebase.com/tools/vugraph_linfetch.php?id=$i"
done
