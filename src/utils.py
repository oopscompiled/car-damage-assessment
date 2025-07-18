import os
from collections import Counter, defaultdict
from glob import glob
import matplotlib.image as imd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

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

        print('\n📊 Initial stats (number of images per original class):')
        for cls in sorted(original_image_counter):
            print(f'Class {cls}: {len(original_image_counter[cls])} images')

        print(f'\n❌ Classes to remove: {self.REMOVE_CLASS}')

    def process_file(self, filepath, images_dir):
        basename = os.path.splitext(os.path.basename(filepath))[0] # removing file format for future smooth usage
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

        # Update class counters (annotation instances)
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


def visualize(images_path, n=5):
    """Visualizes n images from a dataset"""
    data = os.listdir(images_path)

    for _,j in enumerate(data[:n]):
        image = imd.imread(f'{images_path}/{j}')
        plt.imshow(image)
        plt.axis('off')
        plt.show()

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
        target = {target} #remove duplicates
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
                print(f"Skipped (Unknown format): {filename}") # like .avif and others
            except Exception as e:
                print(f"Error while processing {filename}: {e}")