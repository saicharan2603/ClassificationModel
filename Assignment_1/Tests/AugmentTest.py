# importing sys
import sys

# add the path of the Assignment_1 folder to the sys.path
sys.path.append('Assignment_1')

from PreProcessor import Augment
from Utils import Info, GetData

def test_augmented_data(original_data: dict, augmented_data: dict):
    assert augmented_data[b'filenames'] == original_data[b'filenames'] , "Augmented filenames are not correct"
    assert augmented_data[b'batch_label'] == original_data[b'batch_label'] , "Augmented batch label is not correct"

    # testing the labels
    assert augmented_data[b'labels'].shape == (20000,), "Augmented labels shape is not correct"
    assert augmented_data[b'labels'][0].dtype == original_data[b'labels'][0].dtype , "Augmented labels[0] type is not correct"
    assert (augmented_data[b'labels'][0:10000] == original_data[b'labels'][0:10000]).all() or (augmented_data[b'labels'][0:20000: 2] == original_data[b'labels'][0:10000]).all(), "Augmented labels are not correct"

    # testing the data
    assert augmented_data[b'data'].shape == (20000, 3072), "Augmented data shape is not correct"
    assert augmented_data[b'data'][0].shape == original_data[b'data'][0].shape , "Augmented data[0] shape is not correct"
    assert augmented_data[b'data'][0].dtype == original_data[b'data'][0].dtype, "Augmented data type is not correct"
    assert (augmented_data[b'data'][0:10000] == original_data[b'data'][0:10000]).all() or (augmented_data[b'data'][0:20000: 2] == original_data[b'data'][0:10000]).all(), "Augmented data are not correct"

if __name__ == "__main__":
    original_data = GetData.get_train_data()
    augmented_data = Augment.get_Augmented_Data(original_data)

    Info.print_dict_info(augmented_data)
    test_augmented_data(original_data, augmented_data)
