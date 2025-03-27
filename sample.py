import torch
import matplotlib.pyplot as plt
from ddpm import Diffusion
from unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model = UNet().to(device)
#model.load_state_dict(torch.load("models/ddpm_mnist.pth"))
model.load_state_dict(torch.load("models/ddpm_mnist.pth", map_location=torch.device('cpu')))
model.eval()

diffusion = Diffusion(device=device)

@torch.no_grad()
def sample_images(num_images=16):
    x_t = torch.randn((num_images, 1, 28, 28), device=device)
    for t in reversed(range(diffusion.timesteps)):
        t_tensor = torch.tensor([t] * num_images, device=device)
        x_t = diffusion.reverse_process(x_t, t_tensor, model)
    return x_t.cpu()

# 生成并可视化
samples = sample_images(16)
samples = samples.view(16, 28, 28).numpy()

fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axs.flat):
    ax.imshow(samples[i], cmap="gray")
    ax.axis("off")
plt.savefig("a.png")
plt.close()
print()