#!/bin/bash

mkdir ${3}_graphs

for file in $1/*; do
	
	filename=$(basename "$file")
	extension="${filename##*.}"
	filename="${filename%.*}"
	
	mkdir tmp

	sed "/^#include/d" $file -E > tmp/$filename.$2

	joern-parse -o $filename.cpg tmp

	rm -r tmp

	if [ ! -s err ]; then

		joern-export -o tmp --repr all --format graphml $filename.cpg

		if [ ! -s err2 ]; then

			mv tmp/export.graphml ${3}_graphs/$filename.graphml

		fi

		rm -r tmp

	else

		rm $filename.cpg

	fi

done

#python3 process_graph.py graphs

#rm *.cpg