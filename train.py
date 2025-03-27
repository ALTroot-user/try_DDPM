import torch
import torch.optim as optim
from ddpm import Diffusion
from unet import UNet
from utils import get_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据
dataloader = get_dataloader("mnist", batch_size=64)

# 初始化模型和优化器
model = UNet().to(device)
diffusion = Diffusion(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def compute_loss(model, diffusion, x0):
    t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=x0.device)
    xt, noise = diffusion.forward_process(x0, t)
    noise_pred = model(xt, t)
    return torch.nn.functional.mse_loss(noise_pred, noise)
import tqdm
# 训练循环
model.train()
for epoch in tqdm.tqdm(range(10)):
    for x0, _ in dataloader:
        x0 = x0.to(device)
        loss = compute_loss(model, diffusion, x0)
        
        optimizer.zero_grad()
        
        # print(f"Loss: {loss.item()}")
        # print(f"x0 shape: {x0.shape}")
        # print(f"xt shape: {xt.shape}")
        # print(f"t shape: {t.shape}")

        loss.backward()
        optimizer.step()
    
    #print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "models/ddpm_mnist.pth")
