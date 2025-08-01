import os
import random

from dataclasses import dataclass
import torch
from torchinfo import summary

from tqdm import tqdm

from config import Settings, get_settings
from dataset import LoadDatasetProps, load_dataset
from validation.model import AttributeData
from model import Model, ModelProps


random.seed(18022004)


def get_device(use_cuda: bool) -> torch.device:
    if not (use_cuda and torch.cuda.is_available()):
        print("Cuda deactivated. Using cpu")
        return torch.device("cpu")

    print(f"Cuda will use {torch.cuda.get_device_name(0)}")

    return torch.device("cuda")


@dataclass
class TrainIterationProps:
    dataset:    tuple[torch.Tensor, torch.Tensor]
    device:     torch.device
    loss_func:  torch.nn.MSELoss
    model:      Model
    optimizer:  torch.optim.Adam
    save:       str


def iteration(props: TrainIterationProps, iteration_number: int, valid: bool = False):
    props.model.train(not valid)
    
    data: torch.Tensor = props.dataset[0 if not valid else 1]

    loss_result: float = 0

    for image, label in (bar := tqdm(data, desc=f"{iteration_number} iteration")):
        if not valid:
            props.optimizer.zero_grad()
        
        image, label = image.to(props.device), label.to(props.device)
        
        result: torch.Tensor = props.model(image)

        loss = props.loss_func(result, label)

        if not valid:
            loss.backward()

            props.optimizer.step()

        loss_result += float(loss.item()) / len(props.dataset)

        bar.set_description(f"[{str(iteration_number) if not valid else "val"}] loss: {loss_result:.6f}")


def train(iteration_props: TrainIterationProps, epoch: int = 50):
    try:
        iteration(iteration_props, 0, True)

        for index in range(epoch):
            iteration(iteration_props, index + 1)
            print("Saving...")
            torch.save(iteration_props.model.state_dict(), f"{iteration_props.save}_{index + 1}.pth")
            print("Saved!")
            iteration(iteration_props, index + 1, True)
    except KeyboardInterrupt:
        pass
    finally:
        torch.save(iteration_props.model.state_dict(), f"{iteration_props.save}.pth")
        print("Model was saved!")

def main():
    settings: Settings = get_settings()

    device = get_device(True)

    model: Model = Model(ModelProps(
        device,
        4,
        2,
        16
    )).to(device)

    if os.path.exists((save := settings.LOAD)):
        print(f"Loading model from: {save}")
        model.load_state_dict(torch.load(save))
    else:
        print(f"Miss {save}. Creating new model")

    dataset = load_dataset(LoadDatasetProps(
        settings=settings,
        path_to_images="C:/dataset/base/images",
        path_to_json="C:/dataset/base/data.json",
        _class=AttributeData,
        valid_percent=0.2,
    ))

    summary(model, (64, 4, 80, 144))

    train(TrainIterationProps(
        dataset,
        device,
        torch.nn.MSELoss(),
        model,
        torch.optim.RMSprop(model.parameters(), lr=1e-3),
        settings.SAVE,
    ))


if __name__ == "__main__":
    main()
