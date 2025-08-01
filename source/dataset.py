from dataclasses import dataclass
import random
import json
from functools import lru_cache

from PIL import Image
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import Settings
from validation.model import AttributeData


@dataclass(frozen=True)
class GetImageProps:
    dataset_index:  int
    path:           str
    real_index:     int
    transform:      transforms.Compose


@lru_cache
def get_image_data(props: GetImageProps) -> Tensor:
    image = Image.open(f"{props.path}/screenshot_{props.real_index}_{props.dataset_index}.png")
    image = image.resize((144, 80), Image.Resampling.LANCZOS).convert("RGBA")

    return props.transform(image)


@dataclass
class CustomDatasetProps:
    settings:   Settings
    path:       str
    iterations: list[int]


class CustomDataset(Dataset):
    def __init__(self, props: CustomDatasetProps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.iterations = props.iterations
        self.path = props.path

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.iterations)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            get_image_data(GetImageProps(1, self.path, self.iterations[index], self.transform)),
            get_image_data(GetImageProps(2, self.path, self.iterations[index], self.transform)),
        )


@dataclass
class LoadDatasetProps:
    settings:       Settings
    path_to_images: str
    path_to_json:   str
    _class:         type[AttributeData]
    valid_percent:  float


def load_dataset(props: LoadDatasetProps) -> tuple[DataLoader, DataLoader]:
    with open(props.path_to_json) as file:
        data: list[dict] = json.load(file)

    data: list[AttributeData] = [
        props._class.model_validate(dictionary, from_attributes=True) for dictionary in data
    ]

    base_length: int = len(data)

    train_indexes:      list[int] = []
    validation_indexes: list[int] = []

    random.shuffle(data)

    for value in tqdm(data, desc="Processing data..."):
        if (len(validation_indexes) / base_length) < props.valid_percent:
            validation_indexes.append(value.iteration)
            continue

        train_indexes.append(value.iteration)
    
    train_dataset = CustomDataset(CustomDatasetProps(
        props.settings, props.path_to_images, train_indexes
    ))

    validation_dataset = CustomDataset(CustomDatasetProps(
        props.settings, props.path_to_images, validation_indexes
    ))
    
    return (
        DataLoader(train_dataset,       batch_size=64,  shuffle=True,   num_workers=props.settings.THREADS, pin_memory=True),
        DataLoader(validation_dataset,  batch_size=64,  shuffle=False,  num_workers=props.settings.THREADS, pin_memory=True),
    )
