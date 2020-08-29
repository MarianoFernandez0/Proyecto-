DATASET_DIRS=("python_plus_octave/datasets_25_08_2020/dataset_1"
                "python_plus_octave/datasets_25_08_2020/dataset_2"
                "python_plus_octave/datasets_25_08_2020/dataset_3"
                "python_plus_octave/datasets_25_08_2020/dataset_4"
                "python_plus_octave/datasets_25_08_2020/dataset_5"
                "python_plus_octave/datasets_25_08_2020/dataset_6")
RESULTS_FILES=("error_results/datasets_25_08_2020/dataset_1_performance_measures.csv"
                  "error_results/datasets_25_08_2020/dataset_2_performance_measures.csv"
                  "error_results/datasets_25_08_2020/dataset_3_performance_measures.csv"
                  "error_results/datasets_25_08_2020/dataset_4_performance_measures.csv"
                  "error_results/datasets_25_08_2020/dataset_5_performance_measures.csv"
                  "error_results/datasets_25_08_2020/dataset_6_performance_measures.csv")
for index in ${!DATASET_DIRS[*]}; do
  echo ${DATASET_DIRS[$index]}
  echo ${RESULTS_FILES[$index]}
  python compute_errors_urbano.py --dataset_dir ${DATASET_DIRS[$index]} --results_file ${RESULTS_FILES[$index]}
done