import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial.distance import pdist
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut

import json
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def voxel_coords(mask, spacing):
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

def _export_vector_to_slicer(
        mask_center, lv_c, ref_sitk, name,
        path="vector.mrk.json", coord_system="LPS",
):
    D = np.array(ref_sitk.GetDirection()).reshape(3, 3)
    O = np.array(ref_sitk.GetOrigin())

    def to_phys(p_zyx):
        # voxel_coords() returns (z, y, x) * (sz, sy, sx)
        # Reorder to physical-axis order (x, y, z)
        p_xyz = np.asarray(p_zyx)[::-1]

        # Convert to physical coordinates (LPS)
        return O + D @ p_xyz

    p1 = to_phys(lv_c)
    p2 = to_phys(mask_center)

    def cp(i, pos, label):
        return {
            "id": str(i),
            "label": label,
            "position": [float(x) for x in pos],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined",
        }

    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Line",
            "name": name,
            "coordinateSystem": coord_system,
            "controlPoints": [
                cp(1, p1, "lv_c"),
                cp(2, p2, "mask_center"),
            ],
        }],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

class MaskDataAnalyzer:
    def __init__(self, data_dir, patient_ids):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.output_dir = os.path.join(ROOT_DIR, "hms_muscle_analysis_output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.comp_dir = os.path.join(self.output_dir, "comps")
        os.makedirs(self.comp_dir, exist_ok=True)

    def _get_mask_boundary(self, mask):
        mask_e = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
        boundary = mask ^ mask_e
        points = np.argwhere(boundary)

        return points

    def _plot_data(self, output_dir, patient_id, al_val, pm_val, data, xlabel, save_name):
        data = np.asarray(data)
        upper = max(data.max(), al_val, pm_val) * 1.05

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(data, bins=np.linspace(0, upper, 30), color="lightblue", edgecolor="black")
        ax.axvline(al_val, color="tab:blue", lw=2, label=f"AL ({al_val:.2f})")
        ax.axvline(pm_val, color="tab:red", lw=2, label=f"PM ({pm_val:.2f})")
        ax.set_xlim(0, upper)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("# of components")
        ax.set_title(f"{patient_id} (diff components={len(data)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{patient_id}_{save_name}.png"))
        plt.close(fig)

    # center of mass side analysis - only needs 'GT' masks, produces global result
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

            c, _, evecs = pca_axes(lv, spacing)
            pc1 = evecs[:, 0]
            pc1 = orient_pc1(lv, spacing, pc1) # base -> tip direction

            al_c = voxel_coords(al, spacing).mean(axis=0)
            pm_c = voxel_coords(pm, spacing).mean(axis=0)

            pts = voxel_coords(lv, spacing)
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
            sitk.WriteImage(out, os.path.join(self.comp_dir, f"{patient_id}_diff_labeled.nrrd"))

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

                sl = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, n))
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

            self._plot_data(
                output_dir=output_dir,
                patient_id=patient_id,
                al_val=al_frac,
                pm_val=pm_frac,
                data=fracs,
                xlabel="Fraction of voxels with all of their neighbours being LV or muscle",
                save_name="frac_dist"
            )

            all_component_data[patient_id] = {
                "al_component": int(al_component),
                "pm_component": int(pm_component),
                "component_data": curr_component_data
            }

        with open(os.path.join(self.output_dir, "component_frac_data.json"), 'w') as f:
            json.dump(all_component_data, f, indent=3)


    # intrusion analyzer - how 'much' the muscle enters the LV
    def _get_convex_hull_mask_dist(self, mask, convex_hull, spacing, c_crop, comp_c, lv_c, ref_sitk, offset, comp_id=None):
        ch_boundary_p = self._get_mask_boundary(convex_hull)
        
        ch_boundary_mask = np.zeros_like(convex_hull)
        ch_boundary_mask[tuple(ch_boundary_p.T)] = 1

        candidate_points = np.argwhere(mask & ch_boundary_mask) * spacing # world coordinates for distances

        if len(candidate_points) < 2:
            return 0.0
        
        # max dist between any points. might result in a biased distance
        # maybe the max distance is needed between closest point to center
        # and any other point?
        # max_dist = np.max(pdist(candidate_points))

        # idea above - probably not the best?
        d_to_center = np.linalg.norm((candidate_points - c_crop), axis=1)
        min_dist_point = candidate_points[np.argmin(d_to_center)]

        max_dist = np.linalg.norm(candidate_points - min_dist_point, axis=1).max()

        if comp_id is not None:
            max_dist_i = np.argmax(np.linalg.norm(candidate_points - min_dist_point, axis=1))
            rel_p = candidate_points[max_dist_i]

            _export_vector_to_slicer(
                mask_center=comp_c,
                lv_c=lv_c,
                ref_sitk=ref_sitk,
                name=f"{comp_id}_cvec",
                path=f"{comp_id}_cvec.mrk.json"
            )

            _export_vector_to_slicer(
                mask_center=rel_p + offset*spacing,
                lv_c=min_dist_point + offset*spacing,
                ref_sitk=ref_sitk,
                name=f"{comp_id}_mvec",
                path=f"{comp_id}_mvec.mrk.json"
            )

            v1 = (rel_p + offset*spacing) - (min_dist_point + offset*spacing)
            v2 = lv_c - comp_c

            d = np.dot(v1, v2)
            cos = d / (np.linalg.norm(v1) * np.linalg.norm(v2))

            print(f"{comp_id} - {cos}")

        return max_dist

    def _get_edt_max_dist(self, edt_map, mask):
        vals = edt_map[mask]

        if vals.size == 0:
            raise RuntimeError(f"Values for component edt is empty.")

        return float(vals.max() - vals.min())

    def measure_muscle_intrusion(self):
        print(f"Analyzing muscle intrusion data...")
        output_dir = os.path.join(self.output_dir, "intrusion_analysis")
        os.makedirs(output_dir, exist_ok=True)

        all_data = {}

        for patient_id in self.patient_ids:
            print(f"\tProcessing {patient_id}...")

            lv_path = os.path.join(self.data_dir, patient_id, "final_mask_nip.seg.nrrd")
            hull_path = os.path.join(self.data_dir, patient_id, "convex_hull.seg.nrrd")
            al_path = os.path.join(self.data_dir, patient_id, "al_intersection.seg.nrrd")
            pm_path = os.path.join(self.data_dir, patient_id, "pm_intersection.seg.nrrd")
            diff_path = os.path.join(self.data_dir, patient_id, "diff.seg.nrrd")

            lv_sitk, lv = utils.scan_to_np_array(scan_path=lv_path, return_sitk=True)
            lv = lv > 0
            spacing = np.array(lv_sitk.GetSpacing())[::-1]
            lv_c = voxel_coords(lv, spacing).mean(axis=0)

            hull_sitk, hull = utils.scan_to_np_array(scan_path=hull_path, return_sitk=True)
            hull_edt = sitk.SignedMaurerDistanceMap(
                hull_sitk,
                insideIsPositive=True,
                squaredDistance=False,
                useImageSpacing=True
            )
            hull_edt = sitk.GetArrayFromImage(hull_edt)
            
            al = utils.scan_to_np_array(scan_path=al_path)
            pm = utils.scan_to_np_array(scan_path=pm_path)
            diff = utils.scan_to_np_array(scan_path=diff_path)

            diff_labeled, _ = ndimage.label(diff)
            out = sitk.GetImageFromArray(diff_labeled.astype(np.uint16))
            out.CopyInformation(lv_sitk)
            sitk.WriteImage(out, os.path.join(self.comp_dir, f"{patient_id}_diff_labeled.nrrd"))

            components, _ = np.unique(diff_labeled[diff_labeled > 0], return_counts=True)

            slices = ndimage.find_objects(diff_labeled)

            al_component = np.unique(diff_labeled[al > 0])[-1]
            pm_component = np.unique(diff_labeled[pm > 0])[-1]

            al_dist, pm_dist = 0, 0
            dists = []
            curr_data = {}

            for component in components:
                sl = slices[component - 1]
                if sl is None:
                    continue

                sl = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, n))
                            for s, n in zip(sl, diff_labeled.shape))
                offset = np.array([s.start for s in sl])
                
                comp = diff_labeled[sl] == component
                convex_hull = convex_hull_image(comp)

                comp_id = None
                comp_c = None

                # OLD MAX DIST
                max_dist = self._get_convex_hull_mask_dist(
                    mask=comp,
                    convex_hull=convex_hull,
                    spacing=spacing,
                    c_crop = lv_c - offset * spacing,
                    comp_c=comp_c,
                    lv_c=lv_c,
                    ref_sitk=lv_sitk,
                    offset=offset,
                    comp_id = comp_id
                )
                # NEW MAX DIST
                max_dist = self._get_edt_max_dist(
                    edt_map=hull_edt,
                    mask=(diff_labeled == component)
                )

                dists.append(max_dist)
                curr_data[int(component)] = float(max_dist)

                if component == al_component:
                    al_dist = max_dist
                elif component == pm_component:
                    pm_dist = max_dist

            self._plot_data(
                output_dir=output_dir,
                patient_id=patient_id,
                al_val=al_dist,
                pm_val=pm_dist,
                data=dists,
                xlabel="Max distance between any two points of component",
                save_name="max_dist"
            )

            all_data[patient_id] = {
                "al_component": int(al_component),
                "pm_component": int(pm_component),
                "component_data": curr_data
            }

        with open(os.path.join(self.output_dir, "intrusion_data.json"), 'w') as f:
            json.dump(all_data, f, indent=3)

    def _get_json_data(self, path):
        with open(path) as f:
            return json.load(f)

    def _get_norm_pc1_pos(self, c, pc1, t_lo, t_hi, spacing, comp):
        points = voxel_coords(comp, spacing)
        t = ((points - c) @ pc1 - t_lo) / (t_hi - t_lo)
        lo, hi = np.percentile(t, [1, 99])

        # def _norm_z(p_mm):
        #     q = np.atleast_2d(p_mm) - c
        #     return ((q @ pc1) - t_lo) / (t_hi - t_lo)

        return t.mean(), lo, hi

    def create_dataset(self, dataset_name="dataset"):
        print("Creating dataset...")
        dataset = {}

        frac_data = self._get_json_data(os.path.join(self.output_dir, "component_frac_data.json"))
        intrusion_data = self._get_json_data(os.path.join(self.output_dir, "intrusion_data.json"))

        for patient_id in self.patient_ids:
            print(f"\tProcessing {patient_id}...")

            curr_frac_data = frac_data[patient_id]
            curr_intrusion_data = intrusion_data[patient_id]
            al_c = curr_frac_data["al_component"]
            pm_c = curr_frac_data["pm_component"]

            patient_data = {}

            lv_path = os.path.join(self.data_dir, patient_id, "final_mask_nip.seg.nrrd")
            al_path = os.path.join(self.data_dir, patient_id, "al_intersection.seg.nrrd")
            pm_path = os.path.join(self.data_dir, patient_id, "pm_intersection.seg.nrrd")
            diff_path = os.path.join(self.data_dir, patient_id, "diff.seg.nrrd")

            lv_sitk, lv = utils.scan_to_np_array(scan_path=lv_path, return_sitk=True)
            lv = lv > 0
            spacing = np.asarray(lv_sitk.GetSpacing()[::-1], float)
            c, _, evecs = pca_axes(lv, spacing)
            pc1 = evecs[:, 0]
            pc1 = orient_pc1(lv, spacing, pc1) # base -> tip direction
            pts = voxel_coords(lv, spacing)
            t = (pts - c) @ pc1
            t_lo, t_hi = np.percentile(t, [1, 99])

            diff = utils.scan_to_np_array(scan_path=diff_path)

            diff_labeled, _ = ndimage.label(diff)
            components, _ = np.unique(diff_labeled[diff_labeled > 0], return_counts=True)
            slices = ndimage.find_objects(diff_labeled)

            for component in components:
                sl = slices[component - 1]
                if sl is None:
                    continue

                sl = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, n))
                            for s, n in zip(sl, diff_labeled.shape))

                comp = diff_labeled == component
                cropped_comp = diff_labeled[sl] == component

                thickness = ndimage.distance_transform_edt(cropped_comp, sampling=spacing).max()
                rel_vol = np.count_nonzero(comp) / np.count_nonzero(lv)
                norm_mean, norm_lo, norm_hi = self._get_norm_pc1_pos(
                    c=c,
                    pc1=pc1,
                    t_lo=t_lo,
                    t_hi=t_hi,
                    spacing=spacing,
                    comp=comp
                )

                patient_data[int(component)] = {
                    "neighbor_fraction": curr_frac_data["component_data"][str(component)],
                    "intrusion_depth": curr_intrusion_data["component_data"][str(component)],
                    "thickness": float(thickness),
                    "rel_vol": float(rel_vol),
                    "norm_mean": float(norm_mean),
                    "norm_lo": float(norm_lo),
                    "norm_hi": float(norm_hi),
                    "is_muscle": bool(component in (al_c, pm_c))
                }

            dataset[patient_id] = patient_data

        with open(os.path.join(self.output_dir, f"{dataset_name}.json"), 'w') as f:
            json.dump(dataset, f, indent=3)

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
        if patient_id == "patient_0024":
            labeled, n = ndimage.label(diff)
            pm_intersection = labeled == 1
        elif patient_id == "patient_0053":
            labeled, n = ndimage.label(diff)
            pm_intersection = labeled == 1

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

FEATURES = ["neighbor_fraction","intrusion_depth","thickness","rel_vol","norm_mean","norm_lo","norm_hi"]
MODEL_TYPE = "logreg"

class MuscleSegmentationModel:
    def __init__(self, dataset_path: str, data_dir: str):
        self.model = self._init_model()
        self.data_dir = data_dir
        self.dataset_path = dataset_path

        if self.dataset_path.endswith(".json"):
            self.dataset = self._load_json_dataset(self.dataset_path)
        elif self.dataset_path.endswith(".csv"):
            self.dataset = pd.read_csv(self.dataset_path)
        else:
            raise RuntimeError(f"Dataset can only be a .csv or .json file. Got: {dataset_path.split('.')[-1]}")

    def _load_json_dataset(self, dataset_path):
        with open(dataset_path) as f:
            dataset = json.load(f)

        rows = [
            {"patient_id": pid, "component": int(comp), **features}
            for pid, comps in dataset.items()
            for comp, features in comps.items()
        ]

        df = pd.DataFrame(rows)
        df["is_muscle"] = df["is_muscle"].astype(int)

        df.to_csv(os.path.join(os.path.dirname(dataset_path), "dataset.csv"), index=False)

        return df

    def _init_model(self):
        print(f"Using model {MODEL_TYPE} for evaluation!")

        if MODEL_TYPE == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
            )
        elif MODEL_TYPE == "SVM":
            pre = ColumnTransformer(
                [("log", FunctionTransformer(np.log), ["rel_vol"])],
                 remainder='passthrough'
            )

            return make_pipeline(
                pre,
                StandardScaler(),
                SVC(kernel='rbf', C=1.0, class_weight='balanced')
            )
        else:
            raise ValueError(f"Unknow model type: {MODEL_TYPE}")

    def cross_val(self):
        df = self.dataset

        if MODEL_TYPE == "logreg":
            probs = cross_val_predict(
                self.model, df[FEATURES], df["is_muscle"],
                groups=df["patient_id"], cv=LeaveOneGroupOut(),
                method="predict_proba"
            )[:, 1]

            return df.assign(prob=probs)
        elif MODEL_TYPE == "SVM":
            scores = cross_val_predict(
                self.model, df[FEATURES], df["is_muscle"],
                groups=df["patient_id"], cv=LeaveOneGroupOut(),
                method="decision_function"
            )

            return df.assign(score=scores)

    def eval_dataset(self):
        print("Running Leave-One-Out cross validation...")
        result_df = self.cross_val()

        print("Evaluating results...")

        results = {}

        result_col_name = None
        if MODEL_TYPE == "logreg":
            result_col_name = "prob"
        elif MODEL_TYPE == "SVM":
            result_col_name = "score"

        for patient_id in result_df["patient_id"].unique():
            print(f"Evaluating {patient_id} results...")

            lv_path = os.path.join(self.data_dir, patient_id, "final_mask_nip.seg.nrrd")
            lv_sitk, lv = utils.scan_to_np_array(scan_path=lv_path, return_sitk=True)
            lv = lv > 0
            spacing = np.asarray(lv_sitk.GetSpacing()[::-1], float)

            diff_path = os.path.join(self.data_dir, patient_id, "diff.seg.nrrd")
            diff = utils.scan_to_np_array(scan_path=diff_path)
            diff_labeled, _ = ndimage.label(diff)
            components, _ = np.unique(diff_labeled[diff_labeled > 0], return_counts=True)

            c, _, evecs = pca_axes(lv, spacing)
            pc1 = evecs[:, 0]
            pc1 = orient_pc1(lv, spacing, pc1)
            e_y = np.array([0.0, 1.0, 0.0])
            v = e_y - (e_y @ pc1) * pc1
            v /= np.linalg.norm(v)

            w = np.cross(pc1, v)
            w /= np.linalg.norm(w)

            neg_side, pos_side = [], []
            for component in components:
                comp = diff_labeled == component
                comp_c = voxel_coords(comp, spacing).mean(axis=0)

                val = (comp_c - c) @ w
                if val < 0:
                    neg_side.append(component)
                else:
                    pos_side.append(component)

            values = result_df[result_df["patient_id"] == patient_id].set_index("component")[result_col_name]
            picked = [int(values.loc[side].idxmax()) for side in (neg_side, pos_side) if side]

            gt = {int(c) for c in result_df.loc[
                (result_df["patient_id"] == patient_id) & (result_df["is_muscle"] == 1),
                "component"
            ]}
            results[patient_id] = sorted(gt) == sorted(picked)
            print(f"\tpicked: {sorted(picked)} | GT: {sorted(gt)} -> {results[patient_id]}")

        n_correct = sum(results.values())
        print(f"Correct: {n_correct} / {len(results)}")
        print(f"To inspect: {[p for p, ok in results.items() if not ok]}")

        final_res = {}
        final_res["eval_results"] = {
            "dataset": os.path.basename(self.dataset_path),
            "correct": n_correct,
            "all": len(results),
            "accuracy": float(n_correct / len(results)),
            "to_inspect": [p for p, ok in results.items() if not ok],
            "model": MODEL_TYPE,
            "features": FEATURES
        }
        final_res["detailed_results"] = {
            pid: {int(comp): float(s) for comp, s in zip(g["component"], g[result_col_name])}
            for pid, g in result_df.groupby("patient_id")
        }

        with open(os.path.join(os.path.dirname(self.dataset_path), f"{MODEL_TYPE}_results.json"), 'w') as f:
            json.dump(final_res, f, indent=3)


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

    dataset_name = "dataset_edt_dist"

    mda = MaskDataAnalyzer(
        data_dir=data_dir,
        patient_ids=patients_to_process,
    )
    # mda.analyze_sides()
    # mda.analyze_neighbours()
    # mda.measure_muscle_intrusion()
    # mda.create_dataset(dataset_name=dataset_name)

    msm = MuscleSegmentationModel(
        dataset_path=os.path.join(ROOT_DIR, "hms_muscle_analysis_output", f"{dataset_name}.json"),
        data_dir=data_dir
    )
    msm.eval_dataset()

if __name__ == "__main__":
    main()