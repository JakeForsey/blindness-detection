import os

import cv2
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset


class APTOSDataset(TorchDataset):
    def __init__(self, data_frame: pd.DataFrame, data_directory: str, preprocess_pipeline, cache: joblib.Memory):
        super().__init__()
        self._data_frame = data_frame
        self._data_directory = data_directory
        self._preprocess_pipeline = preprocess_pipeline
        self._classes = 5
        self._cached_get_image_diagnosis_id = cache.cache(self._get_image_diagnosis_id)
        self._read_image = True

    def set_read_image(self, value: bool):
        self._read_image = value

    def _load_image(self, id_code):
        # aptos 2015 dataset used .jpeg, aptos 2019 used .png
        for file_extension in [".png", ".jpeg"]:
            file_path = os.path.join(self._data_directory, id_code + file_extension)
            if os.path.exists(file_path):
                break

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _to_w_h_channels(self, image: np.array):
        return image.transpose(2, 0, 1)

    def _get_image_diagnosis_id(self, index: int):
        """
        Get the preprocessed image, diagnosis and id at `index` in the dataset.

        :param index: index of the record to fetch
        :return: Tuple[np.ndarray, np.ndarray, str], preprocessed image, diagnosis, record id
        """

        id_code = self._data_frame["id_code"][index]

        try:
            diagnosis_class = self._data_frame["diagnosis"][index]
            diagnosis = np.array(diagnosis_class, dtype=np.int64)
        except KeyError:
            raise KeyError("Are you using the test (submission) dataset? Try using APTOSSubmissionDataset instead.")

        image = self._load_image(id_code)
        preprocessed_image = self._preprocess_pipeline(image)
        preprocessed_image = self._to_w_h_channels(preprocessed_image)

        return preprocessed_image.astype(np.float32), diagnosis, id_code

    def _get_diagnosis(self, index: int):
        diagnosis_class = self._data_frame["diagnosis"][index]
        diagnosis = np.array(diagnosis_class, dtype=np.int64)

        return diagnosis

    def __getitem__(self, index: int):
        if self._read_image:
            return self._cached_get_image_diagnosis_id(index)
        else:
            return self._get_diagnosis(index)

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
