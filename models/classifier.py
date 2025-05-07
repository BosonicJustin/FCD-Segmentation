import torch
from torch import nn
from monai.networks.nets import resnet50


# Resnet50 modification of BrainIAC
# download weights - https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=r5nyejyo&dl=0
# Rename BrainIAC (1).ckpt to BrainIAC.ckpt
class ResNet50_3D(nn.Module):
    def __init__(self):
        super(ResNet50_3D, self).__init__()

        resnet = resnet50(pretrained=False)  # assuming you're not using a pretrained model
        resnet.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        hidden_dim = resnet.fc.in_features
        self.backbone = resnet
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x


def load_brainiac(checkpoint_path, device='cuda'):
    model = ResNet50_3D()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"]
    filtered_state_dict = {key: value for key, value in state_dict.items() if 'backbone' in key}
    model.load_state_dict(filtered_state_dict)
    
    return model


class FCDDetector(nn.Module):
    def __init__(self, encoder='brainIAC', latent_dim=2048):
        super().__init__()

        if encoder == 'brainIAC':
            self.encoder = load_brainiac('./models/brain_iac_checkpoints/BrainIAC.ckpt')

            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector = nn.Linear(latent_dim, 1)

        # Combine encoder and projector into a single sequential model
        self.classifier = nn.Sequential(
            self.encoder,
            self.projector
        )

    def forward(self, x):
        return self.classifier(x)