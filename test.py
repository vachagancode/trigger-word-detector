import torch

a = torch.randn((1, 1, 4, 4))
# b = (torch.nn.MaxPool2d(kernel_size=3, padding=0, stride=1)(a))

c = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2)(a)

print(a)
print(a.shape)

print(c)
print(c.shape)