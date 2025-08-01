from PIL import Image
import torch
from torchvision import transforms

from tqdm import tqdm
from main import get_device
from model import Model, ModelProps


MODEL_PATH: str = "./saves/another_model_26.pth"

device = get_device(True)

model: Model = Model(ModelProps(
    device,
    4,
    2,
    16
))

model.load_state_dict(torch.load(MODEL_PATH))

model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])


def get_image_path(index: int, dataset_index: int = 1) -> str:
    return f"C:/dataset/base/images/screenshot_{index}_{dataset_index}.png"


def open_image(image_path: str) -> Image.Image:
    return Image.open(image_path)


def convert_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.resize((144, 80), Image.Resampling.LANCZOS).convert("RGBA")

    return transform(image)


def process_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(device)

    return tensor.unsqueeze(0)


def convert_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.cpu()

    img_np = tensor.permute(1, 2, 0).detach().numpy()

    img_np = (img_np * 255).astype("uint8")

    return Image.fromarray(img_np).resize((1152, 648), Image.Resampling.LANCZOS)


base_image_frames:      list[Image.Image] = []
required_image_frames:  list[Image.Image] = []
result_image_frames:    list[Image.Image] = []

for index in (bar := tqdm(range(1000), desc="Main loop")):
    image_path: str = get_image_path(index)
    image: Image.Image = open_image(image_path)

    base_image_frames.append(image)

    required_image_frames.append(open_image(get_image_path(index, 2)))

    tensor: torch.Tensor = process_tensor(convert_to_tensor(image))

    result_tensor = model(tensor)[0]

    result_image_frames.append(convert_to_image(result_tensor))

    bar.set_description(f"{index + 1} / 1000")


def save_as_gif(frames: list[Image.Image], path: str):
    print(f"Saving {path} ...")

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        disposal=2,
        loop=0,
        optimize=True
    )

    print(f"{path} saved!")


save_as_gif(base_image_frames,      "./base.gif")
save_as_gif(required_image_frames,  "./required.gif")
save_as_gif(result_image_frames,    "./result.gif")
