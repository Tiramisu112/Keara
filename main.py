import sys
from modules import data_arrangement as data_arr
from modules import performance_evaluate as perf_ev

root_path = sys.executable[:-23]

train_data, y_true_train = data_arr.pre_process_data(root_path + 'data/KDDTrain_20Percent.txt', 50)
test_data, y_true_test = data_arr.pre_process_data(root_path + 'data/KDDTest.txt', 50)
