#!/bin/bash

mkdir ${2}_graphs

for file in $1/*; do

	filename=$(basename "$file")
	extension="${filename##*.}"
	filename="${filename%.*}"

	if [ "$extension" != "h" ]; then

		mkdir tmp

		if [ "$extension" == "txt" ]; then

			mv $file $1/$filename.cpp

		fi

		g++ $file -O0 -Og -o tmp/$filename.exe

		joern-parse -o $filename.cpg --language ghidra tmp/$filename.exe 2>err 

		rm -r tmp

		if [ true ]; then

			joern-export -o tmp --repr all --format graphml $filename.cpg 2>err2

			if [ ! -s err2 ]; then

				mv tmp/export.graphml ${2}_graphs/$filename.graphml

			fi

			rm -r tmp

		else

			rm $filename.cpg

		fi

	fi

done