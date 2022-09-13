#!/bin/bash

for file in $1/*; do

	filename=$(basename "$file")
	extension="${filename##*.}"
	filename="${filename%.*}"

	python3 unpack_csv.py $file $filename

	mkdir graphs
	cd graphs

	../process_code.sh ../${filename}_positive py ${filename}_positive
	../process_code.sh ../${filename}_negative py ${filename}_negative

	cd ..

done