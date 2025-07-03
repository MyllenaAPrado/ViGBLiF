import json

""" configuration json """


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, "r") as f:
            config = json.loads(f.read())
            return Config(config)


config = Config(
    {
        # optimization
        "batch_size": 2,  # 48 20 2
        "learning_rate": 1e-3,  # 3
        "weight_decay": 1e-4,  # 3
        "n_epoch": 80,

        # data
        "dataset": "Robust",

        # model
        "type": "ViGBLiF",
        "gcn_channels1":64,
        "gcn_channels2":64,
        "emb_dim":128,
        "k_neighboor":12,
        "fcc_1":512,
        "fcc_2":64,
        "svPath": "results",

        # load & save checkpoint
        "model_name": "ViGBLiF",
        "type_name": "ViGBLiF_Robust",
        "ckpt_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/",
    }
)
