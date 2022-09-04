import torch.nn as nn
import torch.nn.functional as F
from .transformer_sn import VisioniTransformer
from .projected_feature import FeatureProjector


class ProjectedViTDiscriminator(nn.Module):
    def __init__(self, vit_dim: int=384, **kwargs):
        super().__init__()
        self.projector = FeatureProjector('efficientnet', (3,256,256), out_dim=1024, expand=True)
        h, w = self.projector.out_shape
        assert h % 16 == 0 and w % 16 == 0
        patch_size=(h//16, w//16)
        self.vit = VisioniTransformer(channels=1024, shape=self.projector.out_shape, patch_size=patch_size, dim=vit_dim, depth=4, heads=3, out_normalize=False, linear_classifier=True)
    
    def forward(self, img):
        _, _, h, w = img.shape
        if h != 256 or w != 256:
            img = F.interpolate(img, (256,256), mode='bilinear')

        projected = self.projector(img)
        out = self.vit(projected)
        return out.view(out.size(0), -1)