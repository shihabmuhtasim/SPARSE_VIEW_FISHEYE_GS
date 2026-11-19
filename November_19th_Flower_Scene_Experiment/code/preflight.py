#!/usr/bin/env python3
"""Validate exact inputs and optionally exercise the real CUDA scene loader."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent.parent
METHODS = ["pca_angle", "angular_fps", "coverage"]
COUNTS = [12, 24, 48, 96]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def validate(source, scene_dir, manifests):
    bases = list(manifests.glob("*_fullview_split_v1.json"))
    if len(bases) != 1:
        raise ValueError("The manifest folder must contain exactly one frozen full-view split")
    base = json.loads(bases[0].read_text())
    tests = base["test_images"]
    candidates = base["train_images"]
    names = sorted(candidates + tests)
    assert candidates == sorted(candidates) and len(set(names)) == len(names)
    assert sorted(tests) == names[::8], "Test list must match the patched repository loader"
    source_hashes = json.loads((PACKAGE / "provenance/training_source_sha256.json").read_text())
    for name, expected in source_hashes.items():
        assert sha(source / name) == expected, "Training source mismatch: " + name
    for name, expected in base["reconstruction"]["model_files_sha256"].items():
        assert sha(scene_dir / "sparse/0" / name) == expected, "COLMAP mismatch: " + name
    for name in names:
        assert Path(name).name == name and (scene_dir / "images" / name).is_file(), "Missing image: " + name
    if base["scene"] == "flowers":
        frozen = json.loads((PACKAGE / "frozen/flowers/flowers_fullview_split_v1.json").read_text())
        assert base["reconstruction"] == frozen["reconstruction"]
        assert candidates == frozen["train_images"] and tests == frozen["test_images"]
        for name, expected in json.loads((PACKAGE / "provenance/image_sha256.json").read_text()).items():
            assert sha(scene_dir / "images" / name) == expected, "Original Flowers image changed: " + name
        geometry = json.loads((PACKAGE / "provenance/flowers_geometry_asset.json").read_text())
        for name, expected in geometry["files_sha256"].items():
            assert sha(scene_dir / "sparse/0" / name) == expected, "Frozen Flowers initialization/cache changed: " + name
    for method in METHODS:
        previous = []
        for count in COUNTS:
            path = manifests / f"{base['scene']}_{method}_{count}_v1.json"
            item = json.loads(path.read_text())
            selected = item["train_images"]
            assert len(selected) == len(set(selected)) == count
            assert set(selected) <= set(candidates) and not set(selected) & set(tests)
            assert item["eligible_train_images"] == candidates and item["test_images"] == tests
            assert item["full_colmap_sha256"] == base["reconstruction"]["model_files_sha256"]
            if method != "pca_angle":
                assert selected[:len(previous)] == previous, "FPS/Coverage must use nested prefixes"
            previous = selected
            if base["scene"] == "flowers":
                frozen = json.loads((PACKAGE / "frozen/flowers" / path.name).read_text())
                assert selected == frozen["train_images"], "Selection differs from executed Flowers run"
    return base


def gpu_loader(source, scene_dir, manifests, gpu, env=None):
    environment = dict(os.environ if env is None else env)
    environment.update(PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # This imports the actual training loader; it performs no optimization.
    program = r'''
import json,os,sys
from pathlib import Path
import torch,numpy as np,PIL
from simple_knn._C import distCUDA2
from scene.dataset_readers import readColmapSceneInfo
from lpipsPyTorch.modules.lpips import LPIPS
assert torch.__version__ == '2.8.0+cu126' and np.__version__ == '2.1.2'
assert PIL.__version__ == '11.0.0'
assert torch.cuda.is_available()
assert torch.isfinite(distCUDA2(torch.randn(16,3,device='cuda'))).all()
net=LPIPS('vgg','0.1');x=torch.zeros(1,3,32,32)
assert abs(net(x,x).item()) < 1e-8
scene=Path(sys.argv[1]);folder=Path(sys.argv[2])
base=json.loads(next(folder.glob('*_fullview_split_v1.json')).read_text())
rows=[];radius=None;points=None
for method in ['pca_angle','angular_fps','coverage']:
 for count in [12,24,48,96]:
  path=folder/f"{base['scene']}_{method}_{count}_v1.json"
  m=json.loads(path.read_text());os.environ['DFGS_FIXED_SPLIT']=str(path)
  data=readColmapSceneInfo(str(scene),'images',True)
  assert [c.image_name for c in data.train_cameras] == m['train_images']
  assert sorted(c.image_name for c in data.test_cameras) == sorted(base['test_images'])
  assert len(data.train_cameras)==count and all(c.camera_model==1 for c in data.train_cameras+data.test_cameras)
  r=float(data.nerf_normalization['radius']);p=len(data.point_cloud.points)
  if radius is None:radius,points=r,p
  assert abs(r-radius)<1e-6 and p==points
  if base['scene']=='flowers':assert p==54865 and abs(r-7.153377532958984)<1e-5
  rows.append({'method':method,'count':count,'test_count':len(base['test_images']),'points':p,'radius':r})
print('\nPREFLIGHT_JSON='+json.dumps({'state':'passed','torch':torch.__version__,'numpy':np.__version__,
 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'loader_audits':rows}))
'''
    subprocess.run([sys.executable, "-c", program, str(scene_dir), str(manifests)],
                   cwd=source, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--with-gpu", action="store_true")
    args = parser.parse_args()
    source, scene, manifests = (p.expanduser().resolve() for p in [args.source, args.scene_dir, args.manifests])
    base = validate(source, scene, manifests)
    print("PASS: source, input identities, all 12 splits, excluded tests and nested selections")
    if args.with_gpu:
        gpu_loader(source, scene, manifests, args.gpu)
    else:
        print("CPU input check only; CUDA extensions, metrics and real loader require --with-gpu")


if __name__ == "__main__":
    main()
