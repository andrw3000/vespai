import os
import json
import numpy as np
import shutil
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

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


def yolov5_polygons_from_json(data_dir, json_files):
    """Extract annotation data as dictionary from JSON file.

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
                'instances': [],
                'filename': image_info['file_name'],
                'image_id': file[:-5] + '-' + image_info['file_name'][4:],
                'image_size': (image_info['height'], image_info['width']),
            }

            for annotation in json_data['annotations']:
                if annotation['image_id'] == image_info['id']:

                    poly = annotation['segmentation']
                    if np.array(poly).shape[0] == 1:
                        xcoords = poly[0][0::2]
                        ycoords = poly[0][1::2]

                    else:
                        xcoords = [pair[0] for pair in poly]
                        ycoords = [pair[1] for pair in poly]

                    instance = {
                        'class': annotation['category_id'] - 1,
                        'coords': [],
                    }
                    for x, y in zip(xcoords, ycoords):
                        instance['coords'].append([x, y])

                    image_dict['instances'].append(instance)

            dict_list.append(image_dict)
        print(os.path.join(data_dir, file))
        move_files_to_folder([os.path.join(data_dir, file)],
                             os.path.join(data_dir, 'archive'))

    # Directory to hold .txt file annotations
    os.makedirs(os.path.join(data_dir, 'ann'), exist_ok=True)

    for image_dict in dict_list:

        print_buffer = []

        # Normalise co-ordinates by the dimensions image_dict['image_size']
        image_height, image_width = image_dict["image_size"]

        # For each instance
        for instance in image_dict['instances']:

            instance_data = '{}'.format(instance['class'])
            for x, y in instance['coords']:
                instance_data += ' {} {}'.format(x / image_width,
                                                 y / image_height,)

            # Write the bbox details to the print_buffer
            print_buffer.append(instance_data)

        # Name of the file which we have to save
        annotation_file = os.path.join(
            data_dir, 'ann',
            image_dict['filename'][4:].replace('jpeg', 'txt'),
        )

        # Save the annotation to disk
        if os.path.exists(os.path.join(data_dir, image_dict['filename'])):
            print('\n'.join(print_buffer), file=open(annotation_file, 'w'))

    return dict_list


def check_polys(label_file, print_labels=False):
    """Plots boxes from label filename in directory arranged as above."""

    with open(label_file, 'r') as file:
        annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(' ') for x in annotation_list]
    # print('Annotations:')
    # for a in annotation_list:
    #     print(list(a))

    # Get the corresponding image file
    image_file = label_file.replace('labels', 'images').replace('txt', 'jpeg')
    assert os.path.exists(image_file)

    # Load the image
    image = Image.open(image_file)
    w, h = image.size  # PIL image dimension: (w, h), whereas cv2: (h, w, c)
    # print('Width, Height = ', image.size)
    plot = ImageDraw.Draw(image, 'RGBA')

    # Class labels
    num_crabro = 0
    num_velutina = 0

    # Exclude the case of no annotations annotation_list = [['']]
    if len(annotation_list[0][0]):
        annotation_list = [[float(y) for y in x] for x in annotation_list]
        for ann in annotation_list:
            ann = np.array(ann)
            ann[1::2] *= w
            ann[2::2] *= h

            # Draw on image
            class_id = ann[0]
            if class_id == 0:
                box_col = 'yellow'  # Vespa crabro
                box_shade = (255, 255, 0, 75)
            elif class_id == 1:
                box_col = 'red'  # Vespa velutina
                box_shade = (255, 0, 0, 75)
            else:
                raise ValueError("Unexpected class_id.")

            ann_list = []
            for x, y in zip(ann[1::2], ann[2::2]):
                ann_list.append((x, y))

            plot.polygon(ann_list, outline=box_col, width=3, fill=box_shade)
            if print_labels:
                plot.text(
                    (np.min(ann[1::2]), np.max(ann[2::2]) + 10),
                    classes[int(class_id)], fill='blue',
                    font=ImageFont.truetype("/Library/Fonts/Arial.ttf", 30),
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


