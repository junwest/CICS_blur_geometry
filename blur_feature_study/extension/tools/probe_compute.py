import sys, numpy as np, cv2
sys.path.insert(0, r"C:\Users\김준서\Desktop\CICS\blur_feature_study")
from src.io_utils import load_config
from src.dataset import load_manifest, load_pair
from src import features as F
from src.psf import make_psf
from src.imaging import apply_blur, wiener_restore, unsharp_mask
cfg = load_config(); man = load_manifest(cfg)
row = man[man.pair_id=="v_calder__1-5"].iloc[0]
pair = load_pair(row, cfg)
cond = [c for c in cfg["blur"]["conditions"] if c["blur_id"]=="disk_r5"][0]
psf = make_psf(cond, cfg); B = apply_blur(pair.tgt, psf); R = wiener_restore(B, psf, 0.005)
mask = F.border_mask(B.shape, 24)
def kp_arr(kps): return np.array([[k.pt[0],k.pt[1],k.size,k.angle,k.response,k.octave,k.class_id] for k in kps])
for det in ["ORB","SIFT"]:
    p = cfg["detectors"][det]["default"]
    for name, img in [("B",B),("R",R),("clean",pair.tgt)]:
        d = F.create_detector(det, p)
        kA, dA = d.detectAndCompute(img, mask)
        d2 = F.create_detector(det, p)
        kD = d2.detect(img, mask)
        for i,k in enumerate(kD): k.class_id = i
        kD_in = kp_arr(kD)
        kC, dC = F.create_detector(det, p).compute(img, list(kD))
        kC_out = kp_arr(kC)
        # cross: compute on other image
        other = R if name=="B" else B
        kX, dX = F.create_detector(det, p).compute(other, list(kD))
        kX_out = kp_arr(kX)
        same_det = np.array_equal(kp_arr(kA)[:,:6], kD_in[:,:6])
        print(det, name, "n_dAc", len(kA), "n_detect", len(kD), "n_compute_same", len(kC), "n_compute_other", len(kX),
              "| detect==detectAndCompute kps:", same_det,
              "| compute ids order preserved:", np.array_equal(kC_out[:,6], np.arange(len(kD))) if len(kC)==len(kD) else "len differs",
              "| attrs unchanged:", np.array_equal(kC_out[:,:6], kD_in[kC_out[:,6].astype(int),:6]),
              "| desc==dAc:", (dA.shape==dC.shape and np.array_equal(dA,dC)),
              "| other ids:", np.array_equal(kX_out[:,6], kC_out[:,6]), "attrs:", np.array_equal(kX_out[:,:6], kC_out[:,:6]))
        if not np.array_equal(kC_out[:,:6], kD_in[kC_out[:,6].astype(int),:6]):
            diff = np.abs(kC_out[:,:6]-kD_in[kC_out[:,6].astype(int),:6]).max(axis=0); print("   max attr diff x,y,size,angle,resp,oct:", diff)
