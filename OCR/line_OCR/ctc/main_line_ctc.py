#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import sys
sys.path.insert(0, '../../..')
import os
from OCR.line_OCR.ctc.trainer_line_ctc import TrainerLineCTC
from OCR.line_OCR.ctc.models_line_ctc import Decoder
from basic.models import FCN_Encoder
from torch.optim import Adam
from basic.generic_dataset_manager import OCRDataset
from basic.random_seed_utils import set_global_seed
import torch.multiprocessing as mp
import torch
import wandb
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_test(rank, params):
    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    # Model trains until max_time_training or max_nb_epochs is reached or KeyboardInterrupt is raised
    try:
        model.train()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    # compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer", "time", "worst_cer"]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)

    if params["wandb"]["use"]:
        wandb.finish()


if __name__ == "__main__":
    dataset_name = "SK"  # ["RIMES", "IAM", "READ_2016", "SK"]
    n_runs = 1  # сколько нейросейтей обучить
    # константы для паддинга
    MIN_HEIGHT = 168  # q75
    MIN_WIDTH = 2533  # q75

    for run_num in range(1, n_runs + 1):
        logger.info(
            "\n" + "-" * 35 + "\n" + "-" * 35 + "\n" +
            f"Run {run_num}/{n_runs}\n" +
            "-" * 35 + "\n" + "-" * 35 + "\n"
        )
        random_seed = run_num * 120
        set_global_seed(random_seed)

        output_folder = f"{dataset_name}_line"
        if n_runs > 1:
            output_folder = os.path.join(output_folder, "ensemble", str(run_num))

        use_transfer_learning = True
        #transfer_learning_checkpoint_path = "outputs/IAM_line/checkpoints/best_1832.pt"
        transfer_learning_checkpoint_path = "outputs/SK_line_v8_500/checkpoints/best_14144.pt"

        params = {
            "dataset_params": {
                "random_seed": random_seed,
                "datasets": {
                    dataset_name: "../../../Datasets/formatted/{}_lines".format(dataset_name),
                },
                "train": {
                    "name": "{}-train".format(dataset_name),
                    "datasets": [dataset_name, ],
                },
                "valid": {
                    "{}-valid".format(dataset_name): [dataset_name, ],
                },
                "dataset_class": OCRDataset,
                "config": {
                    "width_divisor": 8,  # Image width will be divided by 8
                    "height_divisor": 32,  # Image height will be divided by 32
                    "padding_value": 255,  #0  # Image padding value
                    "padding_token": 1000,  # Label padding value (None: default value is chosen)
                    "charset_mode": "CTC",  # add blank token
                    "constraints": [  # Padding for CTC requirements if necessary
                        "CTC_line",
                        "padding",  # всегда добавляем паддинг
                    ],
                    "padding": {
                        "min_height": MIN_HEIGHT,
                        "min_width": MIN_WIDTH
                    },
                    "preprocessings": [
                        # {
                        #     "type": "dpi",  # modify image resolution
                        #     "source": 200,  # from 300 dpi
                        #     "target": 150,  # to 150 dpi
                        # },
                        {
                            "type": "to_RGB",
                            # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                        },
                    ],
                    # Augmentation techniques to use at training time
                    "augmentation": {
                        "dpi": {
                            "proba": 0.2,
                            "min_factor": 0.75,
                            "max_factor": 1.25,
                        },
                        "perspective": {
                            "proba": 0.2,
                            "min_factor": 0,
                            "max_factor": 0.3,
                        },
                        "elastic_distortion": {
                            "proba": 0.2,
                            "max_magnitude": 20,
                            "max_kernel": 3,
                        },
                        "random_transform": {
                            "proba": 0.2,
                            "max_val": 16,
                        },
                        "dilation_erosion": {
                            "proba": 0.2,
                            "min_kernel": 1,
                            "max_kernel": 3,
                            "iterations": 1,
                        },
                        "brightness": {
                            "proba": 0.2,
                            "min_factor": 0.01,
                            "max_factor": 1,
                        },
                        "contrast": {
                            "proba": 0.2,
                            "min_factor": 0.01,
                            "max_factor": 1,
                        },
                        "sign_flipping": {
                            "proba": 0.2,
                        },
                    },
                }
            },

            "model_params": {
                # Model classes to use for each module
                "models": {
                    "encoder": FCN_Encoder,
                    "decoder": Decoder,
                },
                "use_transfer_learning": use_transfer_learning,
                "transfer_learning": {
                    "encoder": ["encoder", transfer_learning_checkpoint_path, True, True],
                    "decoder": ["decoder", transfer_learning_checkpoint_path, True, True],
                },  # dict : {model_name: [state_dict_name, checkpoint_path, learnable, strict], }
                "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
                "dropout": 0.5,  # dropout probability for standard dropout (half dropout probability is taken for spatial dropout)
            },

            "training_params": {
                "output_folder": output_folder,  # folder names for logs and weigths
                "max_nb_epochs": 500000, # 500000,  # max number of epochs for the training
                "max_training_time":  3600 * 24 * 3,  # max training time limit (in seconds)
                "load_epoch": "last",  # ["best", "last"], to load weights from best epoch or last trained epoch
                "interval_save_weights": 250,  # None: keep best and last only
                "use_ddp": False,  # Use DistributedDataParallel
                "use_apex": False,  # Enable mix-precision with apex package
                "nb_gpu": torch.cuda.device_count(),
                "batch_size": 128 if torch.cuda.get_device_name() in ('NVIDIA A100-SXM4-80GB', 'NVIDIA H100 80GB HBM3') else 16,  # mini-batch size per GPU
                "optimizer": {
                    "class": Adam,
                    "args": {
                        "lr": 0.0001,
                        "amsgrad": False,
                    }
                },
                "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
                "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
                "focus_metric": "cer",   # Metrics to focus on to determine best epoch
                "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
                "set_name_focus_metric": "{}-valid".format(dataset_name),
                "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
                "eval_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for evaluation on validation set during training
                "force_cpu": False,  # True for debug purposes to run on cpu only
                "manual_seed": random_seed,
            },
            "wandb": {
                "use": False,
                "project": "SK_hw_recognition",
                "name": "v2"
            },
            "tensorboard": {
                "use": True
            }
        }

        if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
            mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
        else:
            train_and_test(0, params)
