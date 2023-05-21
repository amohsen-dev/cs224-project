#!/bin/bash
for i in $(seq $1 $2)
do
	echo "trimming games $i"
	cat expert_data/games$i.lin | grep md | sed 's/pg||.*/pg||/' > expert_data_trimmed/games$i.lin
done
