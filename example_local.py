import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Patch model paths to use local downloads
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
_original_dinov3_init = None
_original_birefnet_init = None

def _patch_models():
    """Redirect gated HF models to local paths."""
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet

    global _original_dinov3_init, _original_birefnet_init
    _original_dinov3_init = DinoV3FeatureExtractor.__init__
    _original_birefnet_init = BiRefNet.__init__

    def patched_dinov3_init(self, model_name, image_size=512):
        if "dinov3" in model_name:
            local_path = os.path.join(PRETRAINED_DIR, "dinov3")
            if os.path.exists(local_path):
                print(f"[PATCH] Using local dinov3: {local_path}")
                model_name = local_path
        _original_dinov3_init(self, model_name, image_size)

    def patched_birefnet_init(self, model_name="ZhengPeng7/BiRefNet"):
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        if "RMBG" in model_name or "BiRefNet" in model_name:
            local_path = os.path.join(PRETRAINED_DIR, "rmbg2")
            if os.path.exists(local_path):
                print(f"[PATCH] Using local RMBG-2.0: {local_path}")
                model_name = local_path
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=False
        )
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    DinoV3FeatureExtractor.__init__ = patched_dinov3_init
    BiRefNet.__init__ = patched_birefnet_init

_patch_models()

import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Run
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast limit

# 4. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)

# 5. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample.glb", extension_webp=True)
