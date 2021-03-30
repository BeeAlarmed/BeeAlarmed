"""BeeDataset dataset."""
#pylint: disable=invalid-name

import tensorflow as tf
import tensorflow_datasets as tfds
import BeeDataset


class BeeDatasetTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for BeeDataset dataset."""
    DATASET_CLASS = BeeDataset.BeeDataset
    SPLITS = {
        'train': 3  # Number of fake train example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    DL_EXTRACT_RESULT = "./"

if __name__ == '__main__':
    tfds.testing.test_main()
