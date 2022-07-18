import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import transform


class HornetDataset(Dataset):
    """Hornet dataset for classification.
    
    Args:
        image_files: List of JPEG image paths with varying dimensions.
        augmentation: Transform to be applied upon a sample.
    """

    def __init__(self, image_files, augmentation=None):
        super(HornetDataset, self).__init__()
        self.augmentation = augmentation
        self.images = []
        self.targets = []
        for file in image_files:
            image = cv2.imread(os.path.join(file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)

            # One-hot encode targets
            self.targets.append(
                [float(int(file[-6]) == 0), float(int(file[-6]) == 1)]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        sample = [self.images[idx], self.targets[idx]]

        if self.augmentation:
            sample = self.augmentation(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        super(Rescale, self).__init__()
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        resized_image = transform.resize(image, (new_h, new_w))

        return [resized_image, target]


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): If int, square crop is made.
    """

    def __init__(self, output_size):
        super(RandomCrop, self).__init__()
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, target = sample
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        cropped_image = image
        if h > new_h:
            top = np.random.randint(0, h - new_h)
            cropped_image = cropped_image[top: top + new_h, :, :]
        if w > new_w:
            left = np.random.randint(0, w - new_w)
            cropped_image = cropped_image[:, left: left + new_w, :]

        return [cropped_image, target]


class ToTensor(object):
    """Convert ndarray and target (int) in sample to torch.Tensors."""

    def __call__(self, sample):
        image, target = sample

        # Swap color axis
        # numpy image: h x w x c
        # torch image: c x h x w
        image = image.transpose((2, 0, 1))
        return [torch.from_numpy(image).to(dtype=torch.float32),
                torch.as_tensor(target, dtype=torch.float32),
                ]


class MyAugmentations(object):
    """Bespoke image augmentation."""

    def __call__(self, sample):
        image, target = sample
        aug = transforms.Compose([transforms.ColorJitter(.2, .2, .2, .2),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomAutocontrast(p=0.2),
                                  ])
        return aug(image), target


def get_hornet_loader(image_files, batch_size=1, augment=False, shuffle=True):
    """Returns torch data loaders for train/val/test split 80:10:10.

    Args:
        image_files: [list] contains image paths with file names *X.jpeg
        where X, in {0, 1}, is the class number target.
        batch_size: loaded tensors are size (batch_size, c, h, w).
        augment: [bool] appies augmentation for use in training.
        shuffle: [bool] shuffle dataset as received.
    """

    if augment:
        transformations = transforms.Compose(
            [RandomCrop((224, 224)),
             Rescale((256, 256)),
             ToTensor(),
             MyAugmentations(),
             ]
        )
    else:
        transformations = transforms.Compose([Rescale((256, 256)), ToTensor()])

    data_set = HornetDataset(image_files, augmentation=transformations)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    return data_loader
