from dataclasses import dataclass

from torch import nn, device, Tensor, cat


@dataclass
class BlockProps:
    device: device
    input:  int
    output: int
    kernel: int = 3


class Block(nn.Module):
    def __init__(self, params: BlockProps):
        super().__init__()

        self.sequence: nn.Sequential = nn.Sequential(
            nn.Conv2d(params.input, params.output, params.kernel, padding=(params.kernel - 1) // 2),
            nn.BatchNorm2d(params.output),
            nn.ReLU(inplace=True),
            nn.Conv2d(params.output, params.output, params.kernel, padding=(params.kernel - 1) // 2),
            nn.BatchNorm2d(params.output),
            nn.ReLU(inplace=True),
        ).to(params.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.sequence(x)


class DownBlock(nn.Module):
    def __init__(self, params: BlockProps):
        super().__init__()

        self.block: Block = Block(params)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        for_skip: Tensor = self.block(x)
        
        return self.pool(for_skip), for_skip 


class UpBlock(nn.Module):
    def __init__(self, params: BlockProps):
        super().__init__()

        self.pool = nn.ConvTranspose2d(
            params.input,
            params.output,
            kernel_size=2, 
            stride=2
        )

        params.input = params.output * 2

        self.block: Block = Block(params)
    
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        up: Tensor = self.pool(x)
        return self.block(cat([up, skip], dim=1))


@dataclass
class ModelProps:
    device:         device
    input_channels: int
    block_number:   int
    channels_base:  int


class Model(nn.Module):
    def __init__(self, params: ModelProps):
        super().__init__()

        self.down_sequence = nn.Sequential(*[
            DownBlock(BlockProps(
                params.device,
                params.input_channels,
                params.channels_base
            )),

            *[DownBlock(BlockProps(
                params.device,
                params.channels_base * 2 ** index,
                params.channels_base * 2 ** (index + 1)
            )) for index in range(params.block_number - 1)]
        ])

        self.bottleneck = Block(BlockProps(
            params.device,
            params.channels_base * 2 ** (params.block_number - 1),
            params.channels_base * 2 ** params.block_number,
        ))

        self.up_sequence = nn.Sequential(*[
            *[UpBlock(BlockProps(
                params.device,
                params.channels_base * 2 ** index,
                params.channels_base * 2 ** (index - 1)
            )) for index in range(params.block_number, 0, -1)],
        ])

        self.result = nn.Conv2d(
            params.channels_base,
            params.input_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        saved: list[Tensor] = []

        for index, model in enumerate(self.down_sequence.children()):
            x, skip = model(x)
            saved.append(skip)
        
        x = self.bottleneck(x)
        
        for index, model in enumerate(self.up_sequence.children()):
            x = model(x, saved[len(saved) - index - 1])

        return nn.Sigmoid()(self.result(x))


if __name__ == "__main__":
    # DEBUG
    model = Model(ModelProps(
        device("cpu"),
        4,
        1,
        16
    ))
