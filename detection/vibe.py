import numpy as np


class BackgroundSubtractor:
    """Python implementation of ViBe.

    Adapted from https://github.com/232525/ViBe.python
    """

    def __init__(self):
        self.width = 0
        self.height = 0
        self.num_samples = 20
        self.matching_threshold = 20
        self.matching_number = 2
        self.update_factor = 16
        
        # Historical storage
        self.hist_ims = None
        self.hist_buffer = None
        self.last_swapped = 0
        self.num_hist_ims = 2
        
        # Buffers with random values
        self.jump = None
        self.neighbour = None
        self.position = None
        
    def init_history(self, image):

        # Set parameters
        self.height = image.shape[0]
        self.width = image.shape[1]

        # Create the hist_ims
        self.hist_ims = np.zeros(
            (self.height, self.width, self.num_hist_ims), np.uint8,
        )
        for i in range(self.num_hist_ims):
            self.hist_ims[:, :, i] = image
            
        # Create and fill hist_buffer
        self.hist_buffer = np.zeros(
            (self.height,
             self.width,
             self.num_samples - self.num_hist_ims,
             ), np.uint8,
        )
        for i in range(self.num_samples - self.num_hist_ims):
            image_plus_noise = image + np.random.randint(
                -10, 10, (self.height, self.width),
            )
            image_plus_noise[image_plus_noise > 255] = 255
            image_plus_noise[image_plus_noise < 0] = 0
            self.hist_buffer[:, :, i] = image_plus_noise.astype(np.uint8)
        
        # Fill buffers with random values
        max_length = 2 * max(self.height, self.width) + 1
        self.jump = np.zeros((max_length,), np.uint32)
        self.neighbour = np.zeros((max_length,), np.int)
        self.position = np.zeros((max_length,), np.uint32)

        for i in range(max_length):
            self.jump[i] = np.random.randint(1, 2 * self.update_factor + 1)
            self.neighbour[i] = np.random.randint(-1, 2)
            self.neighbour[i] += np.random.randint(-1, 2) * self.width
            self.position[i] = np.random.randint(0, self.num_samples)

    def segmentation(self, image):

        # Initialise segmentation_map
        segmentation_map = np.zeros((self.height, self.width))
        segmentation_map += self.matching_number - 1
        
        # First history image
        delta = np.abs(image - self.hist_ims[:, :, 0])
        mask = delta > self.matching_threshold
        segmentation_map[mask] = self.matching_number
        
        # Remaining hist_ims
        for i in range(1, self.num_hist_ims):
            delta = np.abs(image - self.hist_ims[:, :, i])
            mask = delta <= self.matching_threshold
            segmentation_map[mask] = segmentation_map[mask] - 1
        
        # For swapping
        self.last_swapped = (self.last_swapped + 1) % self.num_hist_ims
        swapping_im_buffer = self.hist_ims[:, :, self.last_swapped]
        
        # Now, we move in the buffer and leave the hist_ims
        num_tests = self.num_samples - self.num_hist_ims
        mask = segmentation_map > 0
        for i in range(num_tests):
            delta_ = np.abs(image - self.hist_buffer[:, :, i])
            mask_ = delta_ <= self.matching_threshold
            mask_ = mask * mask_
            segmentation_map[mask_] = segmentation_map[mask_] - 1
            
            # Swapping: Putting found value in history image buffer
            temp = swapping_im_buffer[mask_].copy()
            swapping_im_buffer[mask_] = self.hist_buffer[:, :, i][mask_].copy()
            self.hist_buffer[:, :, i][mask_] = temp
        
        # simulate the exit inner loop
        mask_ = segmentation_map <= 0
        mask_ = mask * mask_
        segmentation_map[mask_] = 0
        
        # Produces the output. Note that this step is application-dependent
        mask = segmentation_map > 0
        segmentation_map[mask] = 255
        return segmentation_map.astype(np.uint8)
    
    def update(self, image, updating_mask):

        for y in range(1, self.height-1):
            shift = np.random.randint(0, self.width)
            idx = self.jump[shift]
            while idx < self.width - 1:
                index = idx + y * self.width
                if updating_mask[y, idx] == 255:
                    value = image[y, idx]
                    neighbour_id = index + self.neighbour[shift]
                    y_, idx_ = (int(neighbour_id / self.width),
                                int(neighbour_id % self.width),)
                    
                    if self.position[shift] < self.num_hist_ims:
                        self.hist_ims[y, idx, self.position[shift]] = value
                        self.hist_ims[y_, idx_, self.position[shift]] = value
                    else:
                        pos = self.position[shift] - self.num_hist_ims
                        self.hist_buffer[y, idx, pos] = value
                        self.hist_buffer[y_, idx_, pos] = value

                shift = shift + 1
                idx = idx + self.jump[shift]
        
        # First row
        y = 0
        shift = np.random.randint(0, self.width)
        idx = self.jump[shift]
        
        while idx <= self.width - 1:
            # index = idx + y * self.width
            if updating_mask[y, idx] == 0:
                if self.position[shift] < self.num_hist_ims:
                    self.hist_ims[y, idx, self.position[shift]] = image[y, idx]
                else:
                    pos = self.position[shift] - self.num_hist_ims
                    self.hist_buffer[y, idx, pos] = image[y, idx]
            
            shift = shift + 1
            idx = idx + self.jump[shift]
            
        # Last row
        y = self.height - 1
        shift = np.random.randint(0, self.width)
        idx = self.jump[shift]
        
        while idx <= self.width - 1:
            # index = idx + y * self.width
            if updating_mask[y, idx] == 0:
                if self.position[shift] < self.num_hist_ims:
                    self.hist_ims[y, idx, self.position[shift]] = image[y, idx]
                else:
                    pos = self.position[shift] - self.num_hist_ims
                    self.hist_buffer[y, idx, pos] = image[y, idx]
            
            shift = shift + 1
            idx = idx + self.jump[shift]
        
        # First column
        x = 0
        shift = np.random.randint(0, self.height)
        idy = self.jump[shift]
        
        while idy <= self.height - 1:
            # index = x + idy * self.width
            if updating_mask[idy, x] == 0:
                if self.position[shift] < self.num_hist_ims:
                    self.hist_ims[idy, x, self.position[shift]] = image[idy, x]
                else:
                    pos = self.position[shift] - self.num_hist_ims
                    self.hist_buffer[idy, x, pos] = image[idy, x]
            
            shift = shift + 1
            idy = idy + self.jump[shift]
            
        # Last column
        x = self.width - 1
        shift = np.random.randint(0, self.height)
        idy = self.jump[shift]
        
        while idy <= self.height - 1:
            # index = x + idy * self.width
            if updating_mask[idy, x] == 0:
                if self.position[shift] < self.num_hist_ims:
                    self.hist_ims[idy, x, self.position[shift]] = image[idy, x]
                else:
                    pos = self.position[shift] - self.num_hist_ims
                    self.hist_buffer[idy, x, pos] = image[idy, x]
            
            shift = shift + 1
            idy = idy + self.jump[shift]
            
        # The first pixel
        if np.random.randint(0, self.update_factor) == 0:
            if updating_mask[0, 0] == 0:
                position = np.random.randint(0, self.num_samples)
                
                if position < self.num_hist_ims:
                    self.hist_ims[0, 0, position] = image[0, 0]
                else:
                    pos = position - self.num_hist_ims
                    self.hist_buffer[0, 0, pos] = image[0, 0]
