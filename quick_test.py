import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import os, sys
import json
ROOT_DIR = os.path.dirname((os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def voxel_coords(mask, spacing):
    """ Physical coords """
    return np.argwhere(mask) * np.asarray(spacing, dtype=float)

def pca_axes(mask, spacing):
    pts = voxel_coords(mask=mask, spacing=spacing)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]

    return c, evals[order], evecs[:, order]

def orient_pc1(lv, spacing, pc1):
    pts = voxel_coords(lv, spacing)
    ct_c = (np.array(lv.shape) - 1) / 2.0 * spacing
    d = np.linalg.norm(pts - ct_c, axis=1)

    k = max(1, int(0.05 * len(pts)))
    order = np.argsort(d)
    p_near = pts[order[:k]].mean(0)
    p_far = pts[order[-k:]].mean(0)

    margin = (p_far - p_near) @ pc1

    if margin < 0:
        pc1 = -pc1
    
    return pc1

def flag_component(z):
    z_lo, z_hi = np.percentile(z, [2, 98])
    z_mean, z_span = z.mean(), z_hi - z_lo
    reasons = []
    if z_mean < 0.3: reasons.append("too low")
    if z_mean > 0.9: reasons.append("too high")
    if z_span > 0.6: reasons.append("too long")

    return reasons

def _export_axes_to_slicer(
        pc1, w, c, t_lo, t_hi, ref_sitk,
        path="axes.mrk.json", coord_system="LPS",
):
    pc1 = pc1 / np.linalg.norm(pc1)
    w = w / np.linalg.norm(w)
    w_half_len = t_hi + t_lo // 2

    D = np.array(ref_sitk.GetDirection()).reshape(3, 3)
    O = np.array(ref_sitk.GetOrigin())

    def to_phys(p_zyx):
        return O + D @ np.asarray(p_zyx[::-1])

    lines = {
        "pc1": (to_phys(c + t_lo * pc1), to_phys(c + t_hi * pc1)),
        "w": (to_phys(c - w_half_len * w), to_phys(c + w_half_len * w))
    }

    def cp(i, pos):
        return {"id": str(i), "label": f"P{i}", "position": [float(x) for x in pos],
                "selected": True, "locked": False, "visibility": True,
                "positionStatus": "defined"}

    markups = []
    for name, (p1, p2) in lines.items():
        markups.append({
            "type": "Line",
            "name": name,
            "coordinateSystem": coord_system,
            "controlPoints": [cp(1, p1), cp(2, p2)]
        })

    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": markups,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def analyse_muscle_positions(output_dir, ref_sitk, pc1, c, al_c, pm_c, t_lo, t_hi):
    e_y = np.array([0.0, 1.0, 0.0]) # coronal norm vec
    v = e_y - (e_y @ pc1) * pc1
    v /= np.linalg.norm(v)

    w = np.cross(pc1, v)
    w /= np.linalg.norm(w)
    v = w

    _export_axes_to_slicer(
        pc1=pc1, w=w, c=c, t_lo=t_lo, t_hi=t_hi,
        path=os.path.join(output_dir, "axes.mrk.json"),
        ref_sitk=ref_sitk
    )

    def tz_a(p_mm):
        q = np.atleast_2d(p_mm) - c

        z = ((q @ pc1) - t_lo) / (t_hi - t_lo) # norm
        a = q @ v

        return z, a

    centers = [c, al_c, pm_c]
    names = ["LV", "AL", "PM"]
    colors = ["lightgray", "tab:pink", "tab:green"]

    zal, aal = tz_a(al_c)
    zpm, apm = tz_a(pm_c)

    if aal * apm < 0:
        print(f"\tDifferent sides: {aal} - {apm}")
    else:
        print(f"\tSame sides: {aal} - {apm}")

    fig, ax = plt.subplots()

    for center, name, color in zip(centers, names, colors):
        z, a = tz_a(center)
        ax.scatter(
            a, z,
            s=50, c=color,
            label=name
        )

    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("front <-> back")
    ax.set_ylabel("base <-> tip")
    ax.invert_yaxis()
    ax.legend()
    plt.show()
    plt.close(fig=fig)

def main():
    dir = os.path.join(ROOT_DIR, "adaptive_gac_hms_output")

    patient_id = "patient_0015"
    for patient_id in sorted(os.listdir(dir)):
        patient_id = "patient_0003"
        print(f"{patient_id}")
        patient_dir = os.path.join(dir, patient_id)

        lv_sitk, lv = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)
        spacing = np.asarray(lv_sitk.GetSpacing()[::-1], float)
        c, evals, evecs = pca_axes(lv, spacing)
        pc1, pc2, pc3 = evecs[:, 0], evecs[:, 1], evecs[:, 2]
        pc1 = orient_pc1(lv, spacing, pc1)

        muscle_mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "muscle_mask.seg.nrrd"))
        gt_al = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "al_intersection.seg.nrrd"))
        gt_pm = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "pm_intersection.seg.nrrd"))

        al_c = voxel_coords(gt_al, spacing).mean(axis=0)
        pm_c = voxel_coords(gt_pm, spacing).mean(axis=0)
        print(np.sum(gt_al))

        pts = voxel_coords(lv, spacing)
        t = (pts - c) @ pc1
        t_lo, t_hi = np.percentile(t, [1, 99])

        analyse_muscle_positions(
            output_dir=patient_dir,
            ref_sitk=lv_sitk,
            pc1=pc1, c=c,
            al_c=al_c, pm_c=pm_c,
            t_lo=t_lo, t_hi=t_hi
        )
        return

        labeled, _ = ndimage.label(muscle_mask)
        components, counts = np.unique(labeled[labeled > 0], return_counts=True)

        def z_of(p_mm):
            return ((np.atleast_2d(p_mm) - c) @ pc1 - t_lo) / (t_hi - t_lo)

        for comp in components:
            z = z_of(voxel_coords(mask=(labeled == comp), spacing=spacing))
            reasons = flag_component(z)
            print(f"\t{comp}: {', '.join(reasons) if reasons else 'ok'}")

if __name__ == "__main__":
    main()