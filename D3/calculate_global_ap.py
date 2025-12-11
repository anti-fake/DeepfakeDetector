import numpy as np
from sklearn.metrics import average_precision_score


SEED = 418
def set_seed():
    np.random.seed(SEED)

set_seed()

def resample(preds: np.array, labels: np.array, num: int):
    # repeat the test results of different generators to the same number
    current_len = len(preds)
    
    repeat_times = num // current_len
    expanded_preds = np.repeat(preds, repeat_times, axis=0)
    expanded_labels = np.repeat(labels, repeat_times, axis=0)

    rest_num = num % current_len
    indexes = np.random.choice(len(preds), rest_num, replace=False)
    additional_preds = preds[indexes]
    additional_labels = labels[indexes]

    expanded_preds = np.concatenate((expanded_preds, additional_preds), axis=0)
    expanded_labels = np.concatenate((expanded_labels, additional_labels), axis=0)
    # print(len(expanded_labels))
    return expanded_preds, expanded_labels


# "combined_list" is a np list of different generator's testing results, 
# for example: combined_list[i][0], combined_list[i][1] are the detector prediction list and the ground truth list of i-th generator respectively.

# init your combined_list here:

def cal_global_ap(combined_list):

    final_length = max([len(combined_list[i][0]) for i in range(len(combined_list))])
    resampled_combined_list = [resample(combined_list[i][0], combined_list[i][1], final_length) for i in range(len(combined_list))]

    for _ in range(len(combined_list)):
        print(f'{_}: {average_precision_score(resampled_combined_list[_][1], resampled_combined_list[_][0])}')

    final_pred = np.concatenate([resampled_combined_list[i][0] for i in range(len(combined_list))])
    final_true = np.concatenate([resampled_combined_list[i][1] for i in range(len(combined_list))])
    print(f"the average AP:", average_precision_score(final_true, final_pred))

