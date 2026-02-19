# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.

import os
import torch

# print ("torch.cuda.current_device()", torch.cuda.current_device())
# print("torch.cuda.is_available()",  torch.cuda.is_available())





from torch.optim.lr_scheduler import LambdaLR

from daft.cli import HeterogeneousModelFactory, create_parser
from daft.training.hooks import CheckpointSaver, TensorBoardLogger
from daft.training.train_and_eval import train_and_evaluate
import numpy as np

import torchvision

print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")



import random 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

k_of_fold = 10

def main(args=None):
    args = create_parser().parse_args(args=args)
    torch.manual_seed(0)

    current_dir = os.path.dirname(os.path.abspath(__file__)) #/trinity/home/hmo/hmo/ergo_ct/DAFT/
    hdf_dirname = "hdf_cac_128"

    hdf_dir = os.path.join(current_dir, hdf_dirname)

    category = args.category
    modal = args.imgmodal

    print ("category", category)

    for i in range(k_of_fold):
        print ("k=", i)
        # args.train_data = "/trinity/home/hmo/hmo/ergo_ct/DAFT/ergo_train_CHD_128_%s.hdf5" %i 
        # args.val_data = "/trinity/home/hmo/hmo/ergo_ct/DAFT/ergo_val_CHD_128_%s.hdf5" %i 
        # args.test_data = "/trinity/home/hmo/hmo/ergo_ct/DAFT/ergo_test_CHD_128_%s.hdf5" %i 

        args.train_data = os.path.join(hdf_dir, "ergo_train_CHD_128_%s_excluded_%s.hdf5" %(category, i) )
        args.val_data = os.path.join(hdf_dir, "ergo_val_CHD_128_%s_excluded_%s.hdf5" %(category, i) )
        args.test_data = os.path.join(hdf_dir, "ergo_test_CHD_128_%s_excluded_%s.hdf5" %(category, i) )

        ## Add excluded data

        factory = HeterogeneousModelFactory(args)

        experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()

        factory.write_args(experiment_dir / "experiment_args.json")

        train_loader, valid_loader, _ = factory.get_data()
        discriminator = factory.get_and_init_model()
        optimizerD = factory.get_optimizer(filter(lambda p: p.requires_grad, discriminator.parameters()))
        loss = factory.get_loss()

        tb_log_dir = experiment_dir / "tensorboard"
        checkpoints_dir = experiment_dir / "checkpoints"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        train_metrics = factory.get_metrics()
        train_hooks = [TensorBoardLogger(str(tb_log_dir / "train"), train_metrics)]

        eval_metrics_tb = factory.get_metrics()
        eval_hooks = [TensorBoardLogger(str(tb_log_dir / "eval"), eval_metrics_tb)]
        eval_metrics_cp = factory.get_metrics()
        eval_hooks.append(
            CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=1, max_keep=5, metrics=eval_metrics_cp)
        )

        def lr_factor(epoch):
            if epoch <= int(0.8 * args.epoch):
                return 1
            if epoch <= int(0.95 * args.epoch):
                return 0.1
            return 0.05

        # def lr_factor(epoch):
        #     if epoch <= int(0.25 * args.epoch):
        #         return 0.1
        #     if epoch <= int(0.5 * args.epoch):
        #         return 0.05
        #     return 0.05

        # def lr_factor(epoch):
        #     if epoch <= int(0.5 * args.epoch):
        #         return 0.1
        #     if epoch <= int(0.75 * args.epoch):
        #         return 0.05
        #     return 0.05


        # def lr_factor(epoch):
        #     if epoch <= int(0.4 * args.epoch):
        #         return 1
        #     if epoch <= int(0.7 * args.epoch):
        #         return 0.1
        #     return 0.05

        # ## deepsurv
        # def lr_factor(epoch):
        #     if epoch <= int(0.3 * args.epoch):
        #         return 1
        #     if epoch <= int(0.6 * args.epoch):
        #         return 0.1
        #     return 0.05


        scheduler = LambdaLR(optimizerD, lr_lambda=lr_factor)

        # # model_graph = draw_graph(discriminator, input_size=(torch.zeros([1, 1, 30, 128, 128]), torch.zeros([1, 13])), expand_nested=True)
        # model_graph = draw_graph(discriminator, input_size=((1, 1, 30, 128, 128), (1, 13)), expand_nested=True)
        # model_graph.resize_graph(scale=5.0)
        # model_graph.visual_graph.render(format='png')

        # dev = torch.device("cuda")

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print ("dev", dev)

        train_and_evaluate(
            model=discriminator,
            loss=loss,
            train_data=train_loader,
            optimizer=optimizerD,
            scheduler=scheduler,
            num_epochs=args.epoch,
            eval_data=valid_loader,
            train_hooks=train_hooks,
            eval_hooks=eval_hooks,
            device=dev,
            progressbar=False,
            discr_net = args.discriminator_net,
            patience = args.patience,
            outcome = args.outcome,
            fold_i = i,
            category = category,
            modal = modal
        )

    return factory


if __name__ == "__main__":
    main()
