from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import torch
import numpy as np
from PIL import Image
import copy
import json

from graph import open_graph


class LEDDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        image_paths,
        texts,
        seq_lengths,
        mesh_conversions,
        locations,
        viewPoint_location,
        dialogs,
        scan_names,
        levels,
        annotation_ids,
    ):
        self.mode = mode
        self.args = args
        self.image_paths = image_paths
        self.texts = texts
        self.seq_lengths = seq_lengths
        self.mesh_conversions = mesh_conversions
        self.locations = locations
        self.viewPoint_location = viewPoint_location
        self.dialogs = dialogs
        self.scan_names = scan_names
        self.levels = levels
        self.annotation_ids = annotation_ids
        self.mesh2meters = json.load(open(args.mesh2meters))

        if self.args.data_aug and "train" in self.mode:
            print(f"[{self.mode}]: Applying Data Augmentation...")

        self.collect_graphs()

    def collect_graphs(self):  # get scene graphs
        self.scan_graphs = {}
        for scan_id in set(self.scan_names):
            self.scan_graphs[scan_id] = open_graph(self.args.connect_dir, scan_id)

    def augment_data(self, img, location, mesh_conversion):
        img = img.resize(
            (
                int(img.size[0] * self.args.ds_percent),
                int(img.size[1] * self.args.ds_percent),
            )
        )
        location = int(round(location[0] * self.args.ds_percent)), int(
            round((location[1] * self.args.ds_percent))
        )

        if self.args.data_aug and "train" in self.mode:
            # RANDOM CROP MAP
            ##1200*700 --crop by 5%--> 1140*665
            crop_percent = 0.05
            imageWidth, imageHeight = img.size
            widthCrop = imageWidth - (imageWidth * crop_percent)
            heightCrop = imageHeight - (imageHeight * crop_percent)

            cropX, cropY = 1000, 1000

            while (
                widthCrop - 5 < location[1] - cropX
                or location[1] - cropX < 5
                or heightCrop - 5 < location[0] - cropY
                or location[0] - cropY < 5
            ):
                cropX = np.random.randint(0, imageWidth * crop_percent)
                cropY = np.random.randint(0, imageHeight * crop_percent)
            img = img.crop((cropX, cropY, cropX + widthCrop, cropY + heightCrop))
            location = location[0] - cropY, location[1] - cropX

            # ROTATE MAP
            if np.random.randint(2) == 1:
                img = transforms.functional.rotate(img, 180)
                location = int(round((img.size[1] - 1 - location[0]))), int(
                    round(img.size[0] - 1 - location[1])
                )

            # COLOR JITTER
            preprocess = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406, 0.555],
                        std=[0.229, 0.224, 0.225, 0.222],
                    ),
                ]
            )

        else:
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406, 0.555],
                        std=[0.229, 0.224, 0.225, 0.222],
                    ),
                ]
            )

        image = preprocess(img)[:3, :, :]
        return image, location, mesh_conversion

    def gather_all_floors(self, index, image):
        if "train" in self.mode:
            return [], []
        all_maps = torch.zeros(
            self.args.max_floors, image.size()[0], image.size()[1], image.size()[2]
        )
        all_conversions = torch.zeros(self.args.max_floors, 1)
        sn = self.scan_names[index]
        floors = self.mesh2meters[sn].keys()
        for enum, f in enumerate(floors):
            img = Image.open(
                "{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)
            )
            img = img.resize(
                (
                    int(img.size[0] * self.args.ds_percent),
                    int(img.size[1] * self.args.ds_percent),
                )
            )
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406, 0.555],
                        std=[0.229, 0.224, 0.225, 0.222],
                    ),
                ]
            )
            all_maps[enum, :, :, :] = preprocess(img)[:3, :, :]
            all_conversions[enum, :] = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0
        return all_maps, all_conversions

    def get_info(self, index):
        pretty_image = Image.open(self.image_paths[index])
        pretty_image = np.asarray(
            pretty_image.resize(
                (
                    int(
                        pretty_image.size[0] * self.args.ds_percent,
                    ),
                    int(
                        pretty_image.size[1] * self.args.ds_percent,
                    ),
                )
            )
        )[:, :, :3]
        viz_elem = pretty_image

        info_elem = [
            self.dialogs[index],
            self.levels[index],
            self.scan_names[index],
            self.annotation_ids[index],
            self.viewPoint_location[index],
        ]
        return viz_elem, info_elem

    def __getitem__(self, index):
        location = copy.deepcopy(self.locations[index])
        mesh_conversion = self.mesh_conversions[index]
        text = torch.LongTensor(self.texts[index])
        seq_length = np.array(self.seq_lengths[index])
        image, location, mesh_conversion = self.augment_data(
            Image.open(self.image_paths[index]), location, mesh_conversion
        )

        _, imageHeight, imageWidth = image.data.numpy().shape

        gaussian_target = np.zeros((imageHeight, imageWidth))
        gaussian_target[location[0], location[1]] = 1
        gaussian_target = gaussian_filter(
            gaussian_target,
            sigma=self.args.sigma_scalar
            * self.mesh_conversions[index]
            * self.args.ds_percent,
        )
        gaussian_target = torch.tensor(gaussian_target / gaussian_target.sum())

        viz_elem, info_elem = self.get_info(index)

        all_maps, all_conversions = self.gather_all_floors(index, image)

        return (
            viz_elem,
            info_elem,
            image,
            text,
            seq_length,
            gaussian_target,
            mesh_conversion,
            all_maps,
            all_conversions,
        )

    def __len__(self):
        return len(self.image_paths)
