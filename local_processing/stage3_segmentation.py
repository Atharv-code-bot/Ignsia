import os, re, argparse
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import binary_opening, label as scipy_label
import json

CLASS_COLORS = {
    0: (34,  139, 34,  200),   # Canopy → forest green
    1: (150, 150, 150, 200),   # Built  → gray
    2: (210, 180, 140, 200),   # Bare   → tan
}
CLASS_NAMES = {0: "Canopy", 1: "Built", 2: "Bare"}

ENV_BANDS = {
    "NDVI":1,"EVI":2,"NDBI":3,"BSI":4,
    "NDWI":5,"MNDWI":6,"LST_10m":7,
    "B4_red":8,"B8_nir":9,"B11_swir":10
}

def safe_name(ward: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]","_",ward).strip("_")

def read_env_stack(env_tif: str) -> tuple:
    with rasterio.open(env_tif) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile.copy()
    return data, profile

def write_raster(arr, profile, out_path, dtype="float32"):
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    p = profile.copy()
    p.update(count=arr.shape[0], dtype=dtype, compress="lzw",
             tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr.astype(dtype))


def threshold_classify(data: np.ndarray) -> tuple:
    """
    Returns (seg_mask[H,W], confidence[3,H,W])

    seg_mask pixel values: 0=Canopy  1=Built  2=Bare  255=NoData/Water
    confidence: soft probability per class (uniform within threshold class)
    """
    ndvi  = data[ENV_BANDS["NDVI"]-1]
    ndbi  = data[ENV_BANDS["NDBI"]-1]
    mndwi = data[ENV_BANDS["MNDWI"]-1]
    bsi   = data[ENV_BANDS["BSI"]-1]
    H, W  = ndvi.shape

    valid  = np.isfinite(ndvi) & np.isfinite(ndbi)
    water  = (mndwi > 0.10) & valid
    canopy = (ndvi > 0.30) & (ndbi < 0.00) & ~water & valid
    built  = (ndbi > 0.05) & ~canopy & ~water & valid
    bare   = valid & ~canopy & ~built & ~water

    canopy = binary_opening(canopy, iterations=1)
    built  = binary_opening(built,  iterations=1)
    bare   = binary_opening(bare,   iterations=1)

    seg = np.full((H, W), 255, dtype=np.uint8)
    seg[canopy] = 0
    seg[built]  = 1
    seg[bare]   = 2

    conf = np.zeros((3, H, W), dtype=np.float32)
    conf[0][canopy] = 0.85;  conf[1][canopy] = 0.10;  conf[2][canopy] = 0.05
    conf[0][built]  = 0.05;  conf[1][built]  = 0.85;  conf[2][built]  = 0.10
    conf[0][bare]   = 0.05;  conf[1][bare]   = 0.10;  conf[2][bare]   = 0.85

    return seg, conf


def run_segformer(env_tif: str, out_dir: str, ward_name: str,
                  model_checkpoint: str = None) -> np.ndarray:
    
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
    except ImportError:
        print("    [SegFormer] ⚠️  PyTorch/transformers not installed.")
        print("    pip install torch transformers")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    [SegFormer] Using device: {device}")

    with rasterio.open(env_tif) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile.copy()

    B, H, W = data.shape

    for i in range(B):
        vmin, vmax = np.nanpercentile(data[i], 2), np.nanpercentile(data[i], 98)
        if vmax > vmin:
            data[i] = np.clip((data[i] - vmin) / (vmax - vmin), 0, 1)
    data = np.nan_to_num(data, nan=0.0)

    
    if model_checkpoint:
        print(f"    [SegFormer] Loading checkpoint: {model_checkpoint}")
       
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_checkpoint)
    else:
        print(f"    [SegFormer] Initialising SegFormer-B3 with random weights")
        print(f"    ⚠️  For production, load SatlasPretrain weights:")
        print(f"        https://github.com/allenai/satlaspretrain_models")
        config = SegformerConfig(
            num_channels=10,         
            num_labels=3,            
            num_encoder_blocks=4,
            hidden_sizes=[64,128,320,512],
            num_attention_heads=[1,2,5,8],
        )
        model = SegformerForSemanticSegmentation(config)

    model.to(device)
    model.eval()

    tile_sz  = 512
    overlap  = 64
    stride   = tile_sz - overlap

    logits_full = np.zeros((3, H, W), dtype=np.float32)
    weight_full = np.zeros((H, W),    dtype=np.float32)

    import torch.nn.functional as F
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1 = min(y, H - tile_sz);  y2 = y1 + tile_sz
                x1 = min(x, W - tile_sz);  x2 = x1 + tile_sz
                tile = data[:, y1:y2, x1:x2]

                tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                out    = model(pixel_values=tensor)
                logits = out.logits    # [1, 3, H/4, W/4]
                logits = F.interpolate(logits, size=(tile_sz, tile_sz),
                                       mode="bilinear", align_corners=False)
                logits = logits[0].cpu().numpy()   
                
                weight = np.ones((tile_sz, tile_sz), dtype=np.float32)
               
                for i in range(overlap):
                    w = (i + 1) / (overlap + 1)
                    weight[i, :]  *= w;  weight[-(i+1), :] *= w
                    weight[:, i]  *= w;  weight[:, -(i+1)] *= w

                logits_full[:, y1:y2, x1:x2] += logits * weight
                weight_full[y1:y2, x1:x2]    += weight

   
    weight_full = np.maximum(weight_full, 1e-8)
    logits_full /= weight_full

    seg     = logits_full.argmax(axis=0).astype(np.uint8)
    softmax = np.exp(logits_full) / np.exp(logits_full).sum(axis=0, keepdims=True)

    return seg, softmax


def make_rgba_overlay(seg_mask: np.ndarray, profile: dict,
                      out_path: str):
    
    H, W = seg_mask.shape
    rgba = np.zeros((4, H, W), dtype=np.uint8)
    for cls, (r, g, b, a) in CLASS_COLORS.items():
        mask = seg_mask == cls
        rgba[0][mask] = r
        rgba[1][mask] = g
        rgba[2][mask] = b
        rgba[3][mask] = a

    p = profile.copy()
    p.update(count=4, dtype="uint8", compress="lzw",
             photometric="RGBA", tiled=True,
             blockxsize=256, blockysize=256)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(rgba)
    print(f"    → RGBA overlay written: {os.path.basename(out_path)}")



def compute_pixel_class_stats(seg_mask: np.ndarray,
                               ward_name: str) -> dict:
    
    valid   = seg_mask != 255
    total   = valid.sum()
    canopy  = (seg_mask == 0).sum()
    built   = (seg_mask == 1).sum()
    bare    = (seg_mask == 2).sum()

    pixel_area_m2 = 100

    return {
        "ward_name":          ward_name,
        "total_valid_pixels": int(total),
        "total_area_m2":      int(total * pixel_area_m2),
        "canopy_pixels":      int(canopy),
        "built_pixels":       int(built),
        "bare_pixels":        int(bare),
        "canopy_pct":         round(canopy / total * 100, 3) if total > 0 else 0,
        "built_pct":          round(built  / total * 100, 3) if total > 0 else 0,
        "bare_pct":           round(bare   / total * 100, 3) if total > 0 else 0,
        "plantable_area_m2":  int(bare * pixel_area_m2),
        "trees_possible":     int(bare * pixel_area_m2 // 25),
    }



def process_ward_stage3(ward_name: str, stage2_dir: str, out_dir: str,
                         use_segformer: bool = False,
                         model_checkpoint: str = None) -> dict:
    sn      = safe_name(ward_name)
    env_tif = os.path.join(stage2_dir, f"env_stack_{sn}.tif")

    if not os.path.exists(env_tif):
        print(f"  [SKIP] env_stack not found: {env_tif}")
        return None

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  [Stage 3] {ward_name}  mode={'SegFormer' if use_segformer else 'threshold'}")

    data, profile = read_env_stack(env_tif)

    if use_segformer:
        result = run_segformer(env_tif, out_dir, ward_name, model_checkpoint)
        if result is None:
            print("    Falling back to threshold classifier…")
            seg, conf = threshold_classify(data)
        else:
            seg, conf = result
    else:
        seg, conf = threshold_classify(data)

    seg_tif  = os.path.join(out_dir, f"seg_mask_{sn}.tif")
    conf_tif = os.path.join(out_dir, f"seg_confidence_{sn}.tif")
    rgba_tif = os.path.join(out_dir, f"seg_rgba_{sn}.tif")

    write_raster(seg,  profile, seg_tif,  dtype="uint8")
    write_raster(conf, profile, conf_tif, dtype="float32")
    make_rgba_overlay(seg, profile, rgba_tif)

    stats = compute_pixel_class_stats(seg, ward_name)

    stats_path = os.path.join(out_dir, f"seg_stats_{sn}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  [Stage 3] ✅ Canopy={stats['canopy_pct']:.1f}%  "
          f"Built={stats['built_pct']:.1f}%  Bare={stats['bare_pct']:.1f}%  "
          f"Trees≈{stats['trees_possible']}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ward",             default=None)
    parser.add_argument("--all_wards",        action="store_true")
    parser.add_argument("--stage2_dir",       default="./ward_outputs/stage2/")
    parser.add_argument("--out_dir",          default="./ward_outputs/stage3/")
    parser.add_argument("--use_segformer",    action="store_true")
    parser.add_argument("--model_checkpoint", default=None)
    args = parser.parse_args()

    if args.all_wards:
        stacks = [f for f in os.listdir(args.stage2_dir)
                  if f.startswith("env_stack_") and f.endswith(".tif")]
        for s in stacks:
            wn = re.sub(r"^env_stack_","",s).replace(".tif","").replace("_"," ").strip()
            process_ward_stage3(wn, args.stage2_dir, args.out_dir,
                                 args.use_segformer, args.model_checkpoint)
    elif args.ward:
        process_ward_stage3(args.ward, args.stage2_dir, args.out_dir,
                             args.use_segformer, args.model_checkpoint)
    else:
        print("Usage: --ward 'Kothrud'  OR  --all_wards  [--use_segformer]")
