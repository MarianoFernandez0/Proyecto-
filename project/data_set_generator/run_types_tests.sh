#!/bin/bash

files=./configs/*	
iteration=1

for file in $files
do
	echo "Running $file configuration"
	python main_dataset.py "$file"
	mkdir dataset_"$iteration"
	mv -i datasets dataset_"$iteration"
	echo $iteration
	iteration=$((iteration + 1)) 
	mkdir datasets
done

