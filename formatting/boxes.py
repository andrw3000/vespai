import os
import json
import yaml
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Dictionary mapping class names to IDs in `bbox['class']`
classes = {
    0: 'Vespa crabro',
    1: 'Vespa velutina',
}


def move_files_to_folder(list_of_files, destination_folder):
    """Moves files in list to new destination_folder."""

    os.makedirs(destination_folder, exist_ok=True)
    for file in list_of_files:
        try:
            shutil.move(file, destination_folder)
        except BaseException:
            print('Not moved:', file)
            assert False


def remove_empty_dirs(old_dir):
    if not os.listdir(old_dir):
        os.rmdir(old_dir)
    else:
        raise FileExistsError("Some files remain in {}".format(old_dir))


def yolov5_boxes_from_json(data_dir, json_files, box_exp_factor=0.):
    """Extract bounding box data as dictionary from JSON file.

    Our JSON structure f = json.load(open file):
        f.keys() = 'info', 'licenses', 'images', 'annotations', 'categories'
        f['images'][i].keys() = 'id', 'width', 'height', 'file_name',
                                'license', 'date_captured'
        f['annotations'][i].keys() = 'id', 'image_id', 'category_id',
                                     'iscrowd', 'area', 'bbox', 'segmentation'
            Ignore 'iscrowd', 'bbox' and 'segmentation' give [x, y] coords.
        f['categories'][i].keys() = 'supercategory', 'id', 'name'
            'id': 1, 2
            'name': 'Vespa crabro', 'Vespa velutina'

    Args:
        data_dir: [str] dir containing COCO formatted Plainsight exports.
        json_files: Either single string or tuple of strings containing
            chosen JSON files in data_dir.
        box_exp_factor: A fraction of the box size to expand the border.

    Returns:
        dict_list: [List] Dictionaries attached to each image in json_files.
    """
    if isinstance(json_files, str):
        json_files = (json_files,)

    dict_list = []

    print(json_files)
    for file in json_files:

        with open(os.path.join(data_dir, file)) as f:
            json_data = json.load(f)

        for idx, image_info in enumerate(json_data['images']):

            image_dict = {
                'bboxes': [],
                'filename': image_info['file_name'],
                'image_id': file[:-5] + '-' + image_info['file_name'][4:],
                'image_size': (image_info['height'], image_info['width']),
            }

            if json_data['annotations'] is not None:
                for annotation in json_data['annotations']:
                    if annotation['image_id'] == image_info['id']:

                        poly = annotation['segmentation']
                        if np.array(poly).shape[0] == 1:
                            xcoords = poly[0][0::2]
                            ycoords = poly[0][1::2]

                        else:
                            xcoords = [pair[0] for pair in poly]
                            ycoords = [pair[1] for pair in poly]

                        bbox = {
                            'class': annotation['category_id'] - 1,
                            'ymin': np.min(ycoords),
                            'ymax': np.max(ycoords),
                            'xmin': np.min(xcoords),
                            'xmax': np.max(xcoords),
                        }

                        image_dict['bboxes'].append(bbox)

            dict_list.append(image_dict)
        print(os.path.join(data_dir, file))
        move_files_to_folder([os.path.join(data_dir, file)],
                             os.path.join(data_dir, 'archive'))

    # Directory to hold .txt file annotations
    os.makedirs(os.path.join(data_dir, 'ann'), exist_ok=True)

    for image_dict in dict_list:

        print_buffer = []

        # For each bounding box
        for bbox in image_dict['bboxes']:

            # Origin in top-left for YOLOv5
            x_centre = (bbox['xmin'] + bbox['xmax']) / 2
            y_centre = (bbox['ymin'] + bbox['ymax']) / 2
            box_width = (bbox['xmax'] - bbox['xmin'])
            box_height = (bbox['ymax'] - bbox['ymin'])

            # Expand box height/width by a fixed fraction
            box_width *= (1 + box_exp_factor)
            box_height *= (1 + box_exp_factor)

            # Normalise co-ordinates by the dimensions image_dict['image_size']
            image_height, image_width = image_dict["image_size"]
            x_centre /= image_width
            y_centre /= image_height
            box_width /= image_width
            box_height /= image_height

            # Write the bbox details to the print_buffer
            print_buffer.append(
                '{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
                    bbox['class'], x_centre, y_centre, box_width, box_height,
                ))

        # Name of the file which we have to save
        annotation_file = os.path.join(
            data_dir, 'ann',
            image_dict['filename'][4:].replace('jpeg', 'txt'),
        )

        # Save the annotation to disk
        if os.path.exists(os.path.join(data_dir, image_dict['filename'])):
            print('\n'.join(print_buffer), file=open(annotation_file, 'w'))

    return dict_list


def split_train_val_test(data_dir, class_names=None):
    """Split and move files into subsets.

    Args:
        data_dir: [str], dir containing COCO formatted Plainsight exports.
        class_names: (opt) Dictionary relating class_ids to names: print out.
    """

    # Read images and annotations
    image_dir = os.path.join(data_dir, 'img')
    annotation_dir = os.path.join(data_dir, 'ann')
    images = [os.path.join(image_dir, im) for im in
              os.listdir(image_dir) if im[-5:] == '.jpeg']
    annotations = [os.path.join(annotation_dir, an) for an in
                   os.listdir(annotation_dir) if an[-4:] == '.txt']

    images.sort()
    annotations.sort()

    # Split the dataset into train/valid/test splits according to 80:10:10
    train_img, val_img, train_ann, val_ann = train_test_split(
        images, annotations, test_size=0.2, random_state=31, shuffle=True,
    )
    val_img, test_img, val_ann, test_ann = train_test_split(
        val_img, val_ann, test_size=0.5, random_state=31, shuffle=True,
    )

    move_files_to_folder(train_img, os.path.join(data_dir, 'images/train'))
    move_files_to_folder(val_img, os.path.join(data_dir, 'images/val'))
    move_files_to_folder(test_img, os.path.join(data_dir, 'images/test'))
    move_files_to_folder(train_ann, os.path.join(data_dir, 'labels/train'))
    move_files_to_folder(val_ann, os.path.join(data_dir, 'labels/val'))
    move_files_to_folder(test_ann, os.path.join(data_dir, 'labels/test'))

    # Remove empty `img` and `ann` dirs
    shutil.rmtree(image_dir)
    shutil.rmtree(annotation_dir)

    if class_names:

        # Ordered class names list
        class_list = []
        for name in class_names.values():
            class_list.append(name)

        # Save the annotation to disk
        name_file = os.path.join(data_dir, 'obj.names')
        print('\n'.join(class_list), file=open(name_file, 'w'))

        # Data list
        data_list = [
            'classes = {}'.format(len(class_names)),
            'train = {}'.format('images/train'),
            'valid = {}'.format('images/val'),
            'names = {}'.format('obj.names'),
        ]

        # Object data file
        data_file = os.path.join(data_dir, 'obj.data')
        print('\n'.join(data_list), file=open(data_file, 'w'))


def write_yaml(data_dir, yaml_name='dataset.yaml'):
    """Create a YAML file to be read by YOLOv5"""

    data_dict = {
        'train': os.path.join(data_dir, 'images/train/'),
        'val': os.path.join(data_dir, 'images/val/'),
        'test': os.path.join(data_dir, 'images/test/'),
        'nc': 2,
        'names': ['Vespa crabro', 'Vespa velutina'],
    }

    with open(os.path.join(data_dir, yaml_name), 'w') as file:
        yaml.dump(data_dict, file)


def check_bboxes(label_file, print_labels=False):
    """Plots boxes from label filename in directory arranged as above."""

    with open(label_file, 'r') as file:
        annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(' ') for x in annotation_list]
    print('Annotations:')
    for a in annotation_list:
        print(list(a))

    # Get the corresponding image file
    image_file = label_file.replace('labels', 'images').replace('txt', 'jpeg')
    assert os.path.exists(image_file)

    # Load the image
    image = Image.open(image_file)
    w, h = image.size  # PIL image dimension: (w, h), whereas cv2: (h, w, c)
    plot = ImageDraw.Draw(image)

    # Exclude the case of no annotations annotation_list = [['']]
    if len(annotation_list[0][0]):
        annotation_list = [[float(y) for y in x] for x in annotation_list]
        annotations = np.array(annotation_list)

        # Rescale and locate box corners
        new_coords = np.copy(annotations)
        new_coords[:, [1, 3]] = annotations[:, [1, 3]] * w
        new_coords[:, [2, 4]] = annotations[:, [2, 4]] * h
        new_coords[:, 1] = new_coords[:, 1] - (new_coords[:, 3] / 2)
        new_coords[:, 2] = new_coords[:, 2] - (new_coords[:, 4] / 2)
        new_coords[:, 3] = new_coords[:, 1] + new_coords[:, 3]
        new_coords[:, 4] = new_coords[:, 2] + new_coords[:, 4]

        # Draw on image
        num_crabro = 0
        num_velutina = 0
        for ann in new_coords:
            print(len(ann))
            class_id, x0, y0, x1, y1 = ann
            if class_id == 0:
                box_col = 'yellow'  # Vespa crabro
            elif class_id == 1:
                box_col = 'red'  # Vespa velutina
            else:
                raise ValueError("Unexpected class_id.")

            plot.rectangle(((x0, y0), (x1, y1)), outline=box_col, width=4)
            if print_labels:
                plot.text(
                    (x0, y1 + 10), classes[int(class_id)], fill='blue',
                    font=ImageFont.truetype("/Library/Fonts/Arial.ttf", size=30),
                )
            if int(class_id) == 0:
                num_crabro += 1
            elif int(class_id) == 1:
                num_velutina += 1

        print('Vespa crabro (yellow) count: ', num_crabro)
        print('Vespa velutina (red) count: ', num_velutina)

    else:
        print('Vespa crabro (yellow) count: ', 0)
        print('Vespa velutina (red) count: ', 0)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_axis_off()
    ax.imshow(np.array(image))
    fig.tight_layout()
    plt.show()
    return fig, image_file


def plot_from_loader(image, labels, print_labels=False):
    """Plots boxes from output of dataloader.

    Args:
        image: torch.tensor of shape (c, h, w)
        labels: np.ndarray of shape [num_labels, 5] where each label is of
            the form [class_id, x1, y1, x2, y2].
        print_labels: annotate boxes.
    """

    # Load the image
    tensor2pil = transforms.ToPILImage()
    image = tensor2pil(image)
    w, h = image.size  # PIL image dimension: (w, h), whereas cv2: (h, w, c)
    plot = ImageDraw.Draw(image)

    # Rescale and locate box corners
    new_coords = np.copy(labels)
    new_coords[:, [1, 3]] = labels[:, [1, 3]] * w
    new_coords[:, [2, 4]] = labels[:, [2, 4]] * h
    new_coords[:, 1] = new_coords[:, 1] - (new_coords[:, 3] / 2)
    new_coords[:, 2] = new_coords[:, 2] - (new_coords[:, 4] / 2)
    new_coords[:, 3] = new_coords[:, 1] + new_coords[:, 3]
    new_coords[:, 4] = new_coords[:, 2] + new_coords[:, 4]

    # Draw on image
    num_crabro = 0
    num_velutina = 0
    for ann in new_coords:
        class_id, x0, y0, x1, y1 = ann
        if class_id == 0:
            box_col = 'yellow'  # Vespa crabro
        elif class_id == 1:
            box_col = 'red'  # Vespa velutina
        else:
            raise ValueError("Unexpected class_id.")

        plot.rectangle(((x0, y0), (x1, y1)), outline=box_col, width=4)
        if print_labels:
            plot.text(
                (x0, y1 + 10), classes[int(class_id)], fill='blue',
                font=ImageFont.truetype("/Library/Fonts/Arial.ttf", size=30),
            )
        if int(class_id) == 0:
            num_crabro += 1
        elif int(class_id) == 1:
            num_velutina += 1

    print('Vespa crabro (yellow) count: ', num_crabro)
    print('Vespa velutina (red) count: ', num_velutina)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_axis_off()
    ax.imshow(np.array(image))
    fig.tight_layout()
    plt.show()
    return fig, image
