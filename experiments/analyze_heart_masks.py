import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk

import json
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

class MaskDataAnalyzer:
    def __init__(self, data_dir, patient_ids):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.output_dir = os.path.join(ROOT_DIR, "hms_muscle_analysis_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_mask_boundary(self, mask):
        mask_e = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
        boundary = mask ^ mask_e
        points = np.argwhere(boundary)

        return points

    # center of mass side analysis - only needs 'GT' masks, produces global result
    def voxel_coords(self, mask, spacing):
        return np.argwhere(mask) * np.asarray(spacing, dtype=float)

    def pca_axes(self, mask, spacing):
        pts = self.voxel_coords(mask=mask, spacing=spacing)
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]

        return c, evals[order], evecs[:, order]

    def orient_pc1(self, lv, spacing, pc1):
        pts = self.voxel_coords(lv, spacing)
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

    def _are_on_same_slides(self, pc1, c, al_c, pm_c, t_lo, t_hi):
        e_y = np.array([0.0, 1.0, 0.0]) # coronal normal vec
        v = e_y - (e_y @ pc1) * pc1
        v /= np.linalg.norm(v)

        w = np.cross(pc1, v)
        w /= np.linalg.norm(w)

        def tz_a(p_mm):
            q = np.atleast_2d(p_mm) - c

            z = ((q @ pc1) - t_lo) / (t_hi - t_lo)
            a = q @ w

            return z, a

        _, aal = tz_a(al_c)
        _, apm = tz_a(pm_c)

        return aal * apm > 0

    def analyze_sides(self):
        print("Analyzing sides...")

        same_side_ids = []
        diff_side_ids = []

        for patient_id in self.patient_ids:
            print(f"\tProcessing {patient_id}...")

            lv_path = os.path.join(self.data_dir, patient_id, "final_mask_nip.seg.nrrd")
            al_path = os.path.join(self.data_dir, patient_id, "al_intersection.seg.nrrd")
            pm_path = os.path.join(self.data_dir, patient_id, "pm_intersection.seg.nrrd")

            lv_sitk, lv = utils.scan_to_np_array(scan_path=lv_path, return_sitk=True)
            spacing = np.asarray(lv_sitk.GetSpacing()[::-1], float)
            al = utils.scan_to_np_array(scan_path=al_path)
            pm = utils.scan_to_np_array(scan_path=pm_path)

            c, _, evecs = self.pca_axes(lv, spacing)
            pc1 = evecs[:, 0]
            pc1 = self.orient_pc1(lv, spacing, pc1) # base -> tip direction

            al_c = self.voxel_coords(al, spacing).mean(axis=0)
            pm_c = self.voxel_coords(pm, spacing).mean(axis=0)

            pts = self.voxel_coords(lv, spacing)
            t = (pts - c) @ pc1
            t_lo, t_hi = np.percentile(t, [1, 99])

            same_sides = self._are_on_same_slides(
                pc1=pc1,
                c=c,
                al_c=al_c,
                pm_c=pm_c,
                t_lo=t_lo,
                t_hi=t_hi
            )

            if same_sides:
                same_side_ids.append(patient_id)
            else:
                diff_side_ids.append(patient_id)

        data = {
            "same_side": same_side_ids,
            "diff_side": diff_side_ids
        }

        with open(os.path.join(self.output_dir, "side_data.json"), 'w') as f:
            json.dump(data, f, indent=2)

        print(f"All muscles (center of mass) are on different sides: {len(same_side_ids) == 0}")
    
    # diff analyzer - # of voxels for which it's true that adjacent voxels are LV/muscle voxels
    def _crop_mask(self, mask, bbox=None):
        if bbox is None:
            coords = np.argwhere(mask)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1

            mins = np.maximum(mins - 2, 0)
            maxs = np.minimum(maxs + 2, np.array(mask.shape))
            
            z_min, y_min, x_min = mins
            z_max, y_max, x_max = maxs

            return mask[z_min:z_max, y_min:y_max, x_min:x_max], (mins, maxs)
        else:
            mins, maxs = bbox

            z_min, y_min, x_min = mins
            z_max, y_max, x_max = maxs

            return mask[z_min:z_max, y_min:y_max, x_min:x_max]
            
    def _plot_frac_dist(self, output_dir, patient_id, al_frac, pm_frac, fracs):
        fracs = np.asarray(fracs)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(fracs, bins=np.linspace(0, 1, 30), color="lightblue", edgecolor="black")
        ax.axvline(al_frac, color="tab:blue", lw=2, label=f"AL ({al_frac:.2f})")
        ax.axvline(pm_frac, color="tab:red", lw=2, label=f"PM ({pm_frac:.2f})")
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Fraction of voxels with all of their neighbours being LV or muscle")
        ax.set_ylabel(f"# of components")
        ax.set_title(f"{patient_id} (diff components={len(fracs)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{patient_id}_frac_dist.png"))
        plt.close(fig)
    
    def analyze_neighbours(self):
        print(f"Analyzing component (diff) neighbours...")
        output_dir = os.path.join(self.output_dir, "neighbor_analysis")
        os.makedirs(output_dir, exist_ok=True)

        all_component_data = {}

        for patient_id in self.patient_ids:
            print(f"\tProcessing {patient_id}...")

            lv_path = os.path.join(self.data_dir, patient_id, "final_mask_nip.seg.nrrd")
            al_path = os.path.join(self.data_dir, patient_id, "al_intersection.seg.nrrd")
            pm_path = os.path.join(self.data_dir, patient_id, "pm_intersection.seg.nrrd")
            diff_path = os.path.join(self.data_dir, patient_id, "diff.seg.nrrd")

            lv_sitk, lv = utils.scan_to_np_array(scan_path=lv_path, return_sitk=True)
            lv = lv > 0
            al = utils.scan_to_np_array(scan_path=al_path)
            pm = utils.scan_to_np_array(scan_path=pm_path)
            diff = utils.scan_to_np_array(scan_path=diff_path)

            diff_labeled, _ = ndimage.label(diff)
            out = sitk.GetImageFromArray(diff_labeled.astype(np.uint16))
            out.CopyInformation(lv_sitk)
            sitk.WriteImage(out, os.path.join(output_dir, f"{patient_id}_diff_labeled.nrrd"))

            components, _ = np.unique(diff_labeled[diff_labeled > 0], return_counts=True)

            slices = ndimage.find_objects(diff_labeled)
            struct = np.ones((3, 3, 3), dtype=bool)

            al_component = np.unique(diff_labeled[al > 0])[-1]
            pm_component = np.unique(diff_labeled[pm > 0])[-1]

            al_frac, pm_frac = 0, 0
            fracs = []
            curr_component_data = {}

            for component in components:
                sl = slices[component - 1]
                if sl is None:
                    continue

                sl = tuple(slice(max(s.start - 2, 0), min(s.stop + 1, n))
                            for s, n in zip(sl, diff_labeled.shape))
                
                comp = diff_labeled[sl] == component
                allowed = comp | lv[sl]
                eroded = ndimage.binary_erosion(allowed, structure=struct)
                count = np.count_nonzero(comp & eroded)
                size = np.count_nonzero(comp)
                frac = count / size
                fracs.append(frac)
                curr_component_data[int(component)] = float(frac)

                if component == al_component:
                    al_frac = frac
                elif component == pm_component:
                    pm_frac = frac

            self._plot_frac_dist(
                output_dir=output_dir,
                patient_id=patient_id,
                al_frac=al_frac,
                pm_frac=pm_frac,
                fracs=fracs
            )

            all_component_data[patient_id] = {
                "al_component": int(al_component),
                "pm_component": int(pm_component),
                "component_data": curr_component_data
            }

        with open(os.path.join(self.output_dir, "component_frac_data.json"), 'w') as f:
            json.dump(all_component_data, f, indent=3)


    # intrusion analyzer - how 'much' the muscle enters the LV



def _component_touching(diff, struct):
    empty = np.zeros(diff.shape, dtype=bool)
    
    labeled, n = ndimage.label(diff)

    touching = np.unique(labeled[struct])
    touching = touching[touching > 0]
    if touching.size == 0:
        return empty

    sizes = np.bincount(labeled.ravel())
    best = touching[np.argmax(sizes[touching])]

    return labeled == best

def _create_gt_hm_intersection(data_dir, patient_ids):
    temp_dir = os.path.join(data_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    for patient_id in patient_ids:
        print(patient_id)

        al = utils.scan_to_np_array(scan_path=os.path.join(data_dir, patient_id, "papillaryAL.seg.nrrd"))
        pm = utils.scan_to_np_array(scan_path=os.path.join(data_dir, patient_id, "papillaryPM.seg.nrrd"))

        diff = utils.scan_to_np_array(scan_path=os.path.join(data_dir, patient_id, "diff.seg.nrrd"))

        al = al.astype(bool)
        pm = pm.astype(bool)
        diff = diff.astype(bool)

        al_intersection = _component_touching(diff, al)
        pm_intersection = _component_touching(diff, pm)

        if not al.any():
            print(f"Skipping {patient_id}! AL mask is empty.")
            continue

        if not pm.any():
            print(f"Skipping {patient_id}! PM mask is empty.")
            continue

        mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(data_dir, patient_id, "final_mask_nip.seg.nrrd"), return_sitk=True)
        utils.save_data(
            data=al_intersection,
            ref_sitk=mask_sitk,
            output_dir=os.path.join(data_dir, patient_id),
            name="al_intersection",
            is_mask=True,
            color="1.0 0.55 0.1",
            segment_name="al_intersection"
        )

        utils.save_data(
            data=pm_intersection,
            ref_sitk=mask_sitk,
            output_dir=os.path.join(data_dir, patient_id),
            name="pm_intersection",
            is_mask=True,
            color="0.65 0.3 0.9",
            segment_name="pm_intersection"
        )

        # viz for only one run just to verify good components have been picked
        ct = utils.scan_to_np_array(scan_path=os.path.join(data_dir, patient_id, "ct.nii.gz"))

        def _save_slice_with_largest_struct(struct, name):
            counts = struct.sum(axis=(0, 2))
            y = int(np.argmax(counts))

            struct_slice = np.flipud(struct[:, y, :])
            ct_slice = np.flipud(ct[:, y, :])
            mask_slice = np.flipud(mask[:, y, :])

            plt.imshow(ct_slice, cmap='gray')
            plt.imshow(mask_slice, cmap='Reds', alpha=0.3)
            plt.imshow(struct_slice, cmap='Blues', alpha=0.3)
            plt.savefig(os.path.join(temp_dir, f"{patient_id}_{name}.png"))
            plt.close()

        _save_slice_with_largest_struct(al_intersection, "AL")
        _save_slice_with_largest_struct(pm_intersection, "PM")

def main():
    data_dir = os.path.join(ROOT_DIR, "adaptive_gac_hms_output") 

    # if intersection is not yet created
    # patient_ids = sorted(os.listdir(data_dir))
    # _create_gt_hm_intersection(data_dir, patient_ids)

    patients_to_process = [
        "patient_0001",
        "patient_0002",
        "patient_0003",
        "patient_0004",
        "patient_0005",
        "patient_0008",
        "patient_0010",
        "patient_0012",
        "patient_0014",
        "patient_0015",
        "patient_0016",
        "patient_0017",
        "patient_0018",
        "patient_0019",
        "patient_0022",
        "patient_0024",
        "patient_0026",
        "patient_0027",
        "patient_0030",
        "patient_0031",
        "patient_0036",
        "patient_0037",
        "patient_0038",
        "patient_0040",
        "patient_0041",
        "patient_0045",
        "patient_0047",
        "patient_0048",
        "patient_0049",
        "patient_0050",
        "patient_0052",
        "patient_0053"
    ]

    mda = MaskDataAnalyzer(
        data_dir=data_dir,
        patient_ids=patients_to_process,
    )
    # mda.analyze_sides()
    mda.analyze_neighbours()

if __name__ == "__main__":
    main()