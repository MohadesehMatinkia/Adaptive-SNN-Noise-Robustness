import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import matplotlib.pyplot as plt
import numpy as np
import os

# جلوگیری از خطای تکرار کتابخانه
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --- تنظیمات پروژه ---
batch_size = 128
data_path = './data'
num_steps = 25
num_epochs = 3
noise_intensity = 0.5 
initial_beta = 0.9

# --- 1. آماده‌سازی دیتا ---
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

try:
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
except:
    print("Error downloading data.")
    exit()

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True)

# --- 2. ساختار جدید شبکه (اصلاح شده برای رفع خطا) ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
spike_grad = surrogate.fast_sigmoid()

class AdaptiveSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        
        # تغییر مهم: init_hidden=False
        # ما خودمان حافظه را کنترل می‌کنیم تا تداخل گراف پیش نیاید
        self.lif1 = snn.Leaky(beta=initial_beta, spike_grad=spike_grad, 
                              init_hidden=False, learn_beta=True)
        
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=initial_beta, spike_grad=spike_grad, 
                              init_hidden=False, learn_beta=True, output=True)

    def forward(self, x):
        # مقداردهی دستی حافظه (Membrane Potential)
        # این کار باعث می‌شود برای هر بچ، یک حافظه کاملا جدید ساخته شود
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(self.flatten(x))
            # پاس دادن دستی mem1 و دریافت مقدار جدید آن
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            # پاس دادن دستی mem2
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = AdaptiveSNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"Initial Beta: {net.lif1.beta.item():.4f}")

# --- 3. آموزش با تزریق نویز ---
print(f"\nTraining started with Noise Injection (Intensity: {noise_intensity})...")

for epoch in range(num_epochs):
    iter_counter = 0
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        # تزریق نویز
        noise = torch.randn_like(data) * noise_intensity
        noisy_data = torch.clamp(data + noise, 0, 1)

        # فوروارد
        net.train()
        spk_rec, _ = net(noisy_data)

        # محاسبه خطا و بک‌وارد
        loss_val = loss_fn(spk_rec.mean(0), targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        if iter_counter % 100 == 0:
            print(f"Epoch {epoch+1}, Step {iter_counter} \t Loss: {loss_val.item():.4f}")
        iter_counter += 1

print(f"Training Finished!")
print(f"Adapted Beta: {net.lif1.beta.item():.4f}")

# --- 4. تست نهایی و رسم نمودار ---
print("\n--- FINAL ROBUSTNESS TEST ---")
total = 0
correct = 0
sample_vals = {}

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        
        noise = torch.randn_like(data) * noise_intensity
        noisy_data = torch.clamp(data + noise, 0, 1)
        
        spk_rec, mem_rec = net(noisy_data)
        
        _, predicted = spk_rec.mean(0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if not sample_vals:
            sample_vals['data'] = noisy_data[0].cpu()
            sample_vals['spk'] = spk_rec[:, 0, :].cpu()
            sample_vals['mem'] = mem_rec[:, 0, :].cpu()
            sample_vals['pred'] = predicted[0].cpu()
            sample_vals['true'] = targets[0].cpu()

acc = 100 * correct / total
print(f"Accuracy on NOISY data: {acc:.2f}%")

# رسم پلات
print("Showing Plots...")
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Input
ax[0].imshow(sample_vals['data'].reshape(28,28), cmap='gray')
ax[0].set_title(f"Noisy Input\nTrue: {sample_vals['true']} | Pred: {sample_vals['pred']}")
ax[0].axis('off')

# Raster
ax[1].imshow(sample_vals['spk'].T, cmap='Greys', aspect='auto', interpolation='nearest')
ax[1].set_title("Adapted Spike Activity")
ax[1].set_ylabel("Neuron ID")
ax[1].set_xlabel("Time")

# Membrane
for i in range(10):
    if i == sample_vals['true']:
        ax[2].plot(sample_vals['mem'][:, i], label=f"Correct ({i})", linewidth=3, color='green')
    else:
        ax[2].plot(sample_vals['mem'][:, i], alpha=0.15, color='gray')
        
ax[2].set_title("Membrane Potential Dynamics")
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Voltage")
ax[2].legend(loc='upper right')

plt.tight_layout()
plt.show()