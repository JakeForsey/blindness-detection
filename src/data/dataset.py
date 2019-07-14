import os
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset


class APTOSDataset(TorchDataset):
    def __init__(self, data_frame: pd.DataFrame, data_directory: str, preprocess_pipeline):
        super().__init__()
        self._data_frame = data_frame
        self._data_directory = data_directory
        self._preprocess_pipeline = preprocess_pipeline
        self._classes = 5

    def _load_image(self, id_code):
        image = cv2.imread(os.path.join(self._data_directory, id_code + ".png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _to_w_h_channels(self, image: np.array):
        return image.transpose(2, 0, 1)

    @lru_cache(4000)
    def __getitem__(self, index: int, diagnosis_only: bool = False):
        """
        Get the preprocessed image, diagnosis and id at `index` in the dataset.

        :param index: index of the record to fetch
        :param diagnosis_only: bool, if True only fetch the diagnosis (to avoid processing image)
        :return: Tuple[np.ndarray, np.ndarray, str], preprocessed image, diagnosis, record id
        """
        if diagnosis_only:
            return self._data_frame["diagnosis"][index]

        id_code = self._data_frame["id_code"][index]

        image = self._load_image(id_code)
        preprocessed_image = self._preprocess_pipeline(image)
        preprocessed_image = self._to_w_h_channels(preprocessed_image)

        try:
            diagnosis_class = self._data_frame["diagnosis"][index]
        except KeyError:
            raise KeyError("Are you using the test (submission) dataset? Try using APTOSSubmissionDataset instead.")

        return preprocessed_image.astype(np.float32), np.array(diagnosis_class, dtype=np.int64), id_code

    def __len__(self):
        return len(self._data_frame)


class APTOSSubmissionDataset(APTOSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        id_code = self._data_frame["id_code"][index]

        image = self._load_image(id_code)
        preprocessed_image = self._preprocess_pipeline(image)

        preprocessed_image = self._to_w_h_channels(preprocessed_image)

        return preprocessed_image.astype(np.float32)
