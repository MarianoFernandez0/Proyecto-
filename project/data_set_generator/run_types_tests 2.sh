#I/bin/bash
FILES = "./configs/*"
ITERATION = 1
for file in $FILES
do
	echo "Running $file configuration"
	python main_dataset.py "configs/$file"
	mkdir "dataset_$ITERATION"
	mv "datasets dataset_$ITERATION"
done
python main_dataset.py "configs/dataset_1(10Hz).txt"

