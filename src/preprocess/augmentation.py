from typing import List
from typing import Tuple
from typing import Callable
from typing import Union

from albumentations import Compose as AlbumentationsCompose
from torch.utils.data._utils.collate import default_collate

from src.optimization.experiment import AUGMENTATION_STAGES


class AugmentedCollate:
    """
    Leverages the collate_fn of a torch.utils.data.DataLoader to apply image
    augmentations.

    collate_fn pattern is illustrated below:

    class SimpleCustomBatch:
        def __init__(self, data):
            transposed_data = list(zip(*data))
            self.inp = torch.stack(transposed_data[0], 0)
            self.tgt = torch.stack(transposed_data[1], 0)
    
        def pin_memory(self):
            self.inp = self.inp.pin_memory()
            self.tgt = self.tgt.pin_memory()
            return self

    def collate_wrapper(batch):
        return SimpleCustomBatch(batch)
        
    inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    dataset = TensorDataset(inps, tgts)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                        pin_memory=True)

    for batch_ndx, sample in enumerate(loader):
        print(sample.inp.is_pinned())
        print(sample.tgt.is_pinned())
    """
    def __init__(self, stages: Union[List[Tuple[Callable, dict]], List[Tuple[str, dict]]]):
        # If the stages arg has not been initialised
        if isinstance(stages[0][0], str):
            stages = self.initialise_stages(stages)

        self._augmentations = AlbumentationsCompose(
            [augmentation(**kwargs) for augmentation, kwargs in stages]
        )

    def __call__(self, data):
        """
        Data is a list of tuples (image, diagnosis, ids), the result of calling __getitem__() on
        a dataset.

        :param data: data to augment and compile into a batch
        :return: None
        """
        # TODO Handle the case when the input dataset is the submission data set (AKA no diagnoses)
        images, diagnoses, ids = zip(*data)

        augmented_images = [self._augmentations(force_apply=False, image=image.transpose(1, 2, 0))["image"] for image in images]

        # Use the default_collate after augmentation
        return default_collate(
            [(image.transpose(2, 0, 1), diagnosis, id_) for image, diagnosis, id_ in zip(augmented_images, diagnoses, ids)]
        )

    @staticmethod
    def initialise_stages(stages: List[Tuple[str, dict]]) -> List[Tuple[Callable, dict]]:
        return [(AUGMENTATION_STAGES[stage], kwargs) for stage, kwargs in stages]
