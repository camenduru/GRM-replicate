import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/GRM')
os.chdir('/content/GRM')

if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '16'

import torch
from PIL import Image

class Predictor(BasePredictor):
    def setup(self) -> None:
        from webui.runner import GRMRunner
        torch.set_grad_enabled(False)
        device = torch.device('cuda')
        self.runner = GRMRunner(device)
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        model: str = Input(choices=['Zero123++ v1.2', 'Zero123++ v1.1'], default='Zero123++ v1.2'),
        fuse_mesh: bool = True,
        seed: int = Input(default=42),
    ) -> List[Path]:
        input_image_pil = Image.open(input_image)
        run_segmentation = self.runner.run_segmentation(input_image_pil)
        out_gs_vis, out_gs, out_video, out_mesh = self.runner.run_img_to_3d(seed=seed, image=run_segmentation, model=model, fuse_mesh=fuse_mesh, cache_dir='/content/tmp')
        return [Path(out_video), Path(out_mesh), Path(out_gs_vis), Path(out_gs)]