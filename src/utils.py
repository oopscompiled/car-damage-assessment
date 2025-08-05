import os
from collections import Counter, defaultdict
from glob import glob
import matplotlib.image as imd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import numpy as np
import random
import cv2
import matplotlib.patches as patches
from ultralytics import YOLO

SEED = 42

def set_seed(seed=SEED):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

class Preprocess:
    REMOVE_CLASS = 18

    def __init__(self, labels_dirs, image_exts, final_names, global_remap, merge_map):
        self.n_classes = len(set(global_remap.values()))
        self.labels_dirs = labels_dirs
        self.image_exts = image_exts
        self.final_names = final_names
        self.global_remap = global_remap
        self.merge_map = merge_map
        self.final_class_counter = Counter()
        self.removed_labels = 0
        self.removed_images = 0

    def prel_stats(self):
        original_image_counter = defaultdict(set)  # cls -> set of filepaths

        for labels_dir in self.labels_dirs:
            for filename in os.listdir(labels_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(labels_dir, filename)
                    with open(filepath, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                cls = int(parts[0])
                                original_image_counter[cls].add(filepath)

        print('\nInitial stats (number of images per original class):')
        for cls in sorted(original_image_counter):
            print(f'Class {cls}: {len(original_image_counter[cls])} images')

        print(f'\n❌ Classes to remove: {self.REMOVE_CLASS}')

    def process_file(self, filepath, images_dir):
        basename = os.path.splitext(os.path.basename(filepath))[0] # removing file format for smooth usage
        lines_out = []

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls == self.REMOVE_CLASS:
                    continue
                cls = self.merge_map.get(cls, cls)
                cls = self.global_remap.get(cls, None)
                if cls is None:
                    continue
                lines_out.append([cls] + parts[1:])

        for line in lines_out:
            self.final_class_counter[line[0]] += 1

        # If there are valid lines, rewrite the file; otherwise, delete label and image
        if lines_out:
            with open(filepath, 'w') as f:
                for line in lines_out:
                    f.write(' '.join(map(str, line)) + '\n')
        else:
            os.remove(filepath)
            self.removed_labels += 1
            img_removed = False
            for ext in self.image_exts:
                img_path = os.path.join(images_dir, basename + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    self.removed_images += 1
                    img_removed = True
                    break
            if not img_removed:
                print(f'Warning: No image found for {basename} in {images_dir}')

    def process_all(self):
        for labels_dir in self.labels_dirs:
            print(f'\nProcessing {labels_dir}...')
            images_dir = labels_dir.replace('/labels', '/images')
            for filename in os.listdir(labels_dir):
                if filename.endswith('.txt'):
                    self.process_file(os.path.join(labels_dir, filename), images_dir)

        print('\n✅ Done processing.')
        print(f'Removed {self.removed_labels} label files and {self.removed_images} images.')

        print('\nFinal class distribution (instances after merge/remap):')
        for cls_id in range(self.n_classes):
            count = self.final_class_counter.get(cls_id, 0)
            name = self.final_names.get(cls_id, 'unknown')
            print(f'  Class {cls_id} ({name}): {count} instances')
        print(f'\nTotal number of classes with >0 instances: {len(self.final_class_counter)} (expected up to {self.n_classes})')

    def run(self):
        self.prel_stats()
        self.process_all()

class ImagePreprocessor:
    def __init__(self, input_dir, base_output_name='augmented'):
        self.input_dir = input_dir
        self.base_output_name = base_output_name
        self.counter = 0

    def _get_output_dir(self):
        return os.path.join(self.input_dir, f"{self.base_output_name}__{self.counter}")

    def process(self):
        while os.path.exists(self._get_output_dir()):
            self.counter += 1

        output_dir = self._get_output_dir()
        os.makedirs(output_dir)

        for i, filename in enumerate(os.listdir(self.input_dir)):
            filepath = os.path.join(self.input_dir, filename)

            try:
                with Image.open(filepath) as im:
                    im = im.convert("RGB")
                    output_path = os.path.join(output_dir, f"augmented_{i}.jpg")
                    im.save(output_path, "JPEG")
                    print(f"Saved: {output_path}")
            except UnidentifiedImageError:
                print(f"Skipped (Unknown format): {filename}") # like .avif, .heic and others
            except Exception as e:
                print(f"Error while processing {filename}: {e}")


class Inference:
    def __init__(self, model, file_path):
        """
        model: object Detector or Segmenter
        file_path: path to file (image or video)
        """
        self.model = model
        self.file_path = file_path
        self.n_classes = len(self.model.class_names)
        self.image_exts = ['.jpg', '.jpeg', '.webp', '.png']
        self.video_exts = ['.mp4', '.mov', '.avi', '.mkv']
        self.colors = [tuple([random.random() for _ in range(3)]) for _ in range(self.n_classes)]

        self.video_model = YOLO(model.model.ckpt_path) if hasattr(model, "model") else None

    def get_class_color(self, cls_id):
        return self.colors[cls_id % len(self.colors)]

    def visualize_prediction(self, img, results):
        _, ax = plt.subplots(figsize=(12, 8))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        has_masks = "masks" in results and any(m is not None for m in results["masks"])

        for i in range(len(results["boxes"])):
            box = results["boxes"][i]
            cls_id = results["classes"][i]
            confidence = results["confidences"][i]
            x0, y0, x1, y1 = map(int, box)

            label = self.model.class_names[cls_id]
            text = f'{label} {confidence:.2f}'
            color = self.get_class_color(cls_id)

            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.0, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x0, y0, text,
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.7, pad=1, edgecolor='none'),
                color='white',
                fontsize=8
            )

            if has_masks and results["masks"][i] is not None:
                polygon = np.array(results["masks"][i], dtype=np.int32)
                poly_patch = patches.Polygon(
                    polygon, closed=True, facecolor=color, alpha=0.45, edgecolor='none'
                )
                ax.add_patch(poly_patch)

        ax.axis('off')
        ax.set_title('YOLO Predictions + Masks' if has_masks else 'YOLO Predictions')
        plt.tight_layout()
        plt.show()

    def run(self, conf=0.5, iou=0.3, filter_classes=True):
        ext = os.path.splitext(self.file_path)[-1].lower()

        if ext in self.image_exts:
            img = cv2.imread(self.file_path)
            results = self.model.predict(
                img,
                conf=conf,
                iou=iou,
                filter_classes=filter_classes
            )
            self.visualize_prediction(img, results)

        elif ext in self.video_exts:
            model = YOLO(self.model.model.ckpt_path)
            results = model(self.file_path, stream=True, conf=conf, iou=iou, verbose=False)
            out = None
            for result in results:
                annotated_frame = result.plot()
                if out is None:
                    frame_height, frame_width = annotated_frame.shape[:2]
                    out = cv2.VideoWriter(
                        "output.avi", cv2.VideoWriter_fourcc(*"XVID"), 
                        30, (frame_width, frame_height)
                    )
                if annotated_frame.shape[2] == 4:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)
                out.write(annotated_frame)
                cv2.imshow("YOLO Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if out:
                out.release()
            cv2.destroyAllWindows()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

def visualize(images_path, n=5):
    """Visualizes n images from a dataset"""
    data = os.listdir(images_path)

    for _,j in enumerate(data[:n]):
        image = imd.imread(f'{images_path}/{j}')
        plt.imshow(image)
        plt.axis('off')
        plt.show()

def count_instances(label_path):
    """ Counts instances from a label folder"""
    d = dict()

    for filename in os.listdir(label_path):
        if filename.endswith('txt'):
            source = os.path.join(label_path, filename)
            with open(source, 'r') as txt:
                for line in txt:
                    if line.strip():
                        cls = line.split()[0]
                        d[cls] = d.get(cls, 0) + 1
    return d

def manual_inspect(paths: str | list[str], target: int | list[int]):
    """
    Inspects YOLO-style label files and prints matches for given class IDs.

    Args:
        paths (str | list[str]): File path(s) or glob pattern(s) to .txt label files.
        target (int | list[int]): Class ID or list of class IDs to search for.

    Returns:
        str: Summary string with total number of matches.
    """
    if isinstance(paths, str):
        paths = glob(paths)
        
    if isinstance(target, int):
        target = {target} #remove duplicate
    else:
        target = set(target)

    hits = 0

    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                cls = int(line.strip().split()[0])
                if cls in target:
                    print(f'Found class {cls} in {path}')
                    hits += 1

    return f'Total hits: {hits}'

def folder_identity(folder1, folder2, auto_delete=False):
    """
    Compares filenames (without extensions) in two folders and prints
    which files are missing in each. Useful in detecting unlabeled images (backgrounds)

    Optionally: Allows deleting files in folder2 that have no match in folder1.

    Args:
        folder1 (str): Path to the first folder (e.g. images).
        folder2 (str): Path to the second folder (e.g. labels).
        auto_delete (bool): If True, will ask to delete unmatched files in folder2.
    """
    def get_names(folder):
        return set(os.path.splitext(f)[0] for f in os.listdir(folder) if not f.startswith('.'))

    names1 = get_names(folder1)
    names2 = get_names(folder2)

    only_in_1 = names1 - names2
    only_in_2 = names2 - names1

    print(f"🔎 Files in '{folder1}' but missing in '{folder2}':")
    for name in sorted(only_in_1):
        print("  ", name)

    print(f"\n🔎 Files in '{folder2}' but missing in '{folder1}':")
    for name in sorted(only_in_2):
        print("  ", name)

    if only_in_2 and auto_delete:
        confirm = input(f"\n❗ Do you want to delete {len(only_in_2)} unmatched files from '{folder2}'? [Y/N]: ").lower()
        if confirm == "y":
            for filename in os.listdir(folder2):
                name, _ = os.path.splitext(filename)
                if name in only_in_2:
                    path = os.path.join(folder2, filename)
                    os.remove(path)
                    print(f"Deleted: {filename}")
        else:
            print("Deletion canceled.")


# 03_scoring.ipynb
def mask_iou(box, mask):
    """
    IoU between bbox and binary mask
    box: [x0, y0, x1, y1]
    mask: numpy.ndarray (H, W) with 0 and 1
    """
    x0, y0, x1, y1 = map(int, box)
    h, w = mask.shape[:2]

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    mask_crop = mask[y0:y1, x0:x1]

    if mask_crop.size == 0:
        return 0.0

    box_area = (x1 - x0) * (y1 - y0)
    mask_area = np.count_nonzero(mask_crop)
    inter_area = mask_area

    union = float(box_area + np.count_nonzero(mask) - inter_area + 1e-6)
    return inter_area / union if union > 0 else 0


def build_damage_summary(image_path, model_damage, model_parts, conf=0.65, iou_thr=0.1):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}\nPlease check image path")

    damage_results = model_damage.predict(img, conf=conf, iou=iou_thr)
    parts_results = model_parts.predict(img, conf=conf, iou=iou_thr)

    damage_summary = []

    for i, dmg_box in enumerate(damage_results["boxes"]):
        damage_cls = model_damage.class_names[damage_results["classes"][i]]
        damage_conf = damage_results["confidences"][i]

        best_match = None
        best_score = 0

        for j, _ in enumerate(parts_results["boxes"]):
            part_cls = model_parts.class_names[parts_results["classes"][j]]

            if "masks" in parts_results and parts_results["masks"][j] is not None:
                # polygon to binary mask
                polygon = np.array(parts_results["masks"][j], dtype=np.int32)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 1)
                score = mask_iou(dmg_box, mask)
            else:
                score = mask_iou(dmg_box, np.zeros(img.shape[:2], dtype=np.uint8))


            if score > best_score:
                best_score = score
                best_match = part_cls

        if best_score > 0.3:  # 
            damage_summary.append({
                "damage": damage_cls,
                "part": best_match,
                "confidence": damage_conf,
                "iou": best_score
            })

    return damage_summary

def text_prepare(text):
    if text is None:
        raise ValueError("You need to pass text (list of strings or text)")
    
    cwd = os.getcwd()
    os.makedirs('report', exist_ok=True)
    folder = os.path.join(cwd, 'report')
    file_path = os.path.join(folder, 'report.txt')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        if isinstance(text, str):
            f.write(text)
        else:
            for line in text:
                f.write(str(line) + '\n')

    print('File created')