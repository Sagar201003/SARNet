import torch

ckpt = torch.load('results/epochs/best_model.pth', map_location='cpu')

inference_ckpt = {
    'G_AB': ckpt['G_AB'],
    'D_B': ckpt['D_B']
}

torch.save(inference_ckpt, 'results/epochs/inference_model.pth')
print("Saved inference_model.pth")
