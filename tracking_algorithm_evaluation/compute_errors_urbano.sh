DATASET_DIRS=("python_plus_octave/datasets/dataset_1"
                "python_plus_octave/datasets/dataset_2"
                "python_plus_octave/datasets/dataset_3"
                "python_plus_octave/datasets/dataset_4"
                "python_plus_octave/datasets/dataset_5"
                "python_plus_octave/datasets/dataset_6")
RESULTS_FILES=("error_results/urbano/dataset_1_performance_measures.csv"
                  "error_results/urbano/dataset_2_performance_measures.csv"
                  "error_results/urbano/dataset_3_performance_measures.csv"
                  "error_results/urbano/dataset_4_performance_measures.csv"
                  "error_results/urbano/dataset_5_performance_measures.csv"
                  "error_results/urbano/dataset_6_performance_measures.csv")
for index in ${!DATASET_DIRS[*]}; do
  echo ${DATASET_DIRS[$index]}
  echo ${RESULTS_FILES[$index]}
  python compute_errors_urbano.py --dataset_dir ${DATASET_DIRS[$index]} --results_file ${RESULTS_FILES[$index]}
done