import os

from utils.data_utils import FedChexMimicDataset


# @DATASET_REGISTRY.register()
class FedChexMimic():
    dataset_dir = "fedchexmimic"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 2

        federated_train_x = []
        federated_test_x = []
        for net_id in range(cfg.DATASET.USERS):
            train_set = FedChexMimicDataset(
                base_path=self.dataset_dir, 
                site=net_id+1, 
                attribute_type=cfg.DATASET.ATTRIBUTE_TYPE, 
                attributes=cfg.DATASET.ATTRIBUTES,
                modality_type=cfg.DATASET.MODALITY_TYPE,
                resolution=224, 
                depth=3, 
                train=True, 
                # transform=None
            )

            test_set = FedChexMimicDataset(
                base_path=self.dataset_dir, 
                site=net_id+1, 
                attribute_type=cfg.DATASET.ATTRIBUTE_TYPE,
                attributes=cfg.DATASET.ATTRIBUTES,
                modality_type=cfg.DATASET.MODALITY_TYPE,
                resolution=224, 
                depth=3, 
                train=False, 
                # transform=None
            )

            federated_train_x.append(train_set)
            federated_test_x.append(test_set)

        self.federated_train_x = federated_train_x
        self.federated_test_x = federated_test_x
        self.lab2cname = {'NOT Pleural Effusion': 0, 'Pleural Effusion': 1}
        self.classnames = {'NOT Pleural Effusion', 'Pleural Effusion'}