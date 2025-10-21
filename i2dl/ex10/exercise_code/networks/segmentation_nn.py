"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        self.num_classes = num_classes
        
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        # --- Encoder (Feature Extractor) ---
        
        # Use pretrained MobileNetV2 as encoder - much smaller than ResNet-18
        # MobileNetV2 is designed to be lightweight while maintaining good performance
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Extract feature extraction layers from MobileNetV2
        # Let's use fewer layers to reduce parameters and match dimensions properly
        self.encoder = nn.ModuleList([
            # Stage 1: Input processing (3 -> 16 channels)
            nn.Sequential(
                mobilenet.features[0],  # Conv2d(3, 32) + BatchNorm + ReLU6
                mobilenet.features[1],  # InvertedResidual: 32 -> 16
            ),  # Output: 16 channels, 120x120
            
            # Stage 2: First downsampling (16 -> 24 channels)
            nn.Sequential(
                mobilenet.features[2],  # InvertedResidual: 16 -> 24
                mobilenet.features[3],  # InvertedResidual: 24 -> 24
            ),  # Output: 24 channels, 60x60
            
            # Stage 3: Second downsampling (24 -> 32 channels)
            nn.Sequential(
                mobilenet.features[4],  # InvertedResidual: 24 -> 32
                mobilenet.features[5],  # InvertedResidual: 32 -> 32
                mobilenet.features[6],  # InvertedResidual: 32 -> 32
            ),  # Output: 32 channels, 30x30
            
            # Stage 4: Third downsampling (32 -> 64 channels)
            nn.Sequential(
                mobilenet.features[7],  # InvertedResidual: 32 -> 64
                mobilenet.features[8],  # InvertedResidual: 64 -> 64
                mobilenet.features[9],  # InvertedResidual: 64 -> 64
                mobilenet.features[10], # InvertedResidual: 64 -> 64
            ),  # Output: 64 channels, 15x15
        ])

        # --- Decoder (upsampling and skip connections) ---
        
        # Optimized decoder with correct channel dimensions
        
        # Upsampling stage 3 (64 -> 32)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_block3 = ConvLayer(32 + 32, 32)  # 32 from upconv + 32 from skip
        
        # Upsampling stage 2 (32 -> 24)
        self.upconv2 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.conv_block2 = ConvLayer(24 + 24, 24)  # 24 from upconv + 24 from skip
        
        # Upsampling stage 1 (24 -> 16)
        self.upconv1 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.conv_block1 = ConvLayer(16 + 16, 16)  # 16 from upconv + 16 from skip
        
        # Final upsampling and classification
        self.final_upconv = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, self.num_classes, kernel_size=1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #  
        ########################################################################
        
        # Store skip connections
        skip_connections = []
        
        # Encoder (downsampling) with skip connections
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            skip_connections.append(x)  # Store all encoder outputs for skip connections
        
        # Decoder (upsampling) with skip connections
        # Start from the deepest layer (bottleneck)
        
        # Upsampling stage 3 (64 -> 32)
        x = self.upconv3(x)
        skip = skip_connections[-2]  # Stage 3 output (32 channels)
        # Match dimensions by interpolating if necessary
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block3(x)
        
        # Upsampling stage 2 (32 -> 24)
        x = self.upconv2(x)
        skip = skip_connections[-3]  # Stage 2 output (24 channels)
        # Match dimensions by interpolating if necessary
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block2(x)
        
        # Upsampling stage 1 (24 -> 16)
        x = self.upconv1(x)
        skip = skip_connections[-4]  # Stage 1 output (16 channels)
        # Match dimensions by interpolating if necessary
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block1(x)
        
        # Final upsampling and classification - ensure output matches input size
        x = self.final_upconv(x)
        # Make sure final output is 240x240
        if x.shape[-2:] != (240, 240):
            x = F.interpolate(x, size=(240, 240), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

# if __name__ == "__main__":
#     from torchinfo import summary
#     summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")