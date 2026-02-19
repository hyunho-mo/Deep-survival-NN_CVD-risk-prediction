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
from itertools import chain
from operator import itemgetter
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
import os 
import matplotlib.pyplot as plt

from ..models.base import BaseModel, check_is_unique
from .hooks import Hook
from .wrappers import DataLoaderWrapper
import numpy as np
import random 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# modal = "fullct"
# # modal = "heart"
# # modal = "cac"

trained_model_dir = os.path.join(parent_dir, 'trained_model')
if not os.path.exists(trained_model_dir):
    os.makedirs(trained_model_dir)

figure_dir = os.path.join(parent_dir, 'Figure')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

def train_and_evaluate(
    model: BaseModel,
    loss: BaseModel,
    train_data: DataLoaderWrapper,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    num_epochs: int = 1,
    eval_data: Optional[DataLoaderWrapper] = None,
    train_hooks: Optional[Sequence[Hook]] = None,
    eval_hooks: Optional[Sequence[Hook]] = None,
    device: Optional[torch.device] = None,
    progressbar: bool = True,
    discr_net: str = "resnet",
    patience: int = 10,
    outcome: str = "CHD",
    fold_i: int = 0,
    category: str = "cacs0",
    modal: str = "cac",
) -> None:
    """Train and evaluate a model.

    Evaluation is run after every epoch.

    Args:
      model (BaseModel):
        Instance of model to call.
      loss (BaseModel):
        Instance of loss to compute.
      train_data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from for training.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      optimizer (Optimizer):
        Instance of Optimizer to use.
      scheduler (_LRScheduler):
        Optional; Scheduler to adjust the learning rate.
      num_epochs (int):
        Optional; For how many epochs to train.
      eval_data (DataLoaderWrapper):
        Optional; Instance of DataLoader to obtain batches from for evaluation.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      train_hooks (list of Hook):
        Optional; List of hooks to call during training.
      eval_hooks (list of Hook):
        Optional; List of hooks to call during evaluation.
      device (torch.device):
        Optional; Which device to run on.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """
    train_hooks = train_hooks or []
    train_hooks = list(train_hooks)
    evaluator = None
    if eval_data is not None:
        evaluator = ModelEvaluator(
            model=model, loss=loss, data=eval_data, device=device, hooks=eval_hooks, progressbar=progressbar,
        )

    trainer = ModelTrainer(
        model=model,
        loss=loss,
        data=train_data,
        optimizer=optimizer,
        device=device,
        hooks=train_hooks,
        progressbar=progressbar,
    )

    get_lr = itemgetter("lr")
    train_loss_hist = []
    eval_loss_hist = []
    train_cindex_hist = []
    eval_cindex_hist = []
    best_val_loss = 1000000
    best_cindex = 0
    flag = 0
    flag_loss = 0
    flag_cindex = 0
    stopping_index = False 
    for i in range(num_epochs):
        lr = [get_lr(pg) for pg in optimizer.param_groups]
        print("EPOCH: {:>3d};    Learning rate: {}".format(i, lr))
        # train_outputs, model = trainer.run()
        train_loss, train_cindex = trainer.run()

        # train_outputs = trainer.run()        
        # train_loss = train_outputs['total_loss'].detach().cpu().numpy()

        if evaluator is not None:
            
            val_loss, val_cindex = evaluator.run()            
            # eval_outputs = evaluator.run()            
            # val_loss = eval_outputs['total_loss'].detach().cpu().numpy()

            print ("Train_loss - %.4f / Validation_loss - %.4f" %(train_loss,  val_loss))
            print ("Train_cindex - %.4f / Validation_cindex - %.4f" %(train_cindex,  val_cindex))

            train_loss_hist.append(train_loss)
            eval_loss_hist.append(val_loss)
            train_cindex_hist.append(train_cindex)
            eval_cindex_hist.append(val_cindex)

            # torch.save(model, os.path.join(trained_model_dir, '%s_trained_network_%s_%s.pth' %(discr_net, outcome, fold_i)))

            # if discr_net == "resnet" or discr_net == "cnn2fc":
            # if discr_net == "cnn2fc":
            # Early stopping based on val_loss

            # if i >= 10:




            # #############
            # if best_val_loss > val_loss:
            #     best_val_loss = val_loss
            #     flag = 0
            #     torch.save(model, os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s.pth' %(discr_net, modal, outcome, fold_i)))
            #     print ("Model saved")
            # else:
            #     flag += 1         
            #     print ("flag: ", flag)           
            #     if flag >= patience:                        
            #         print ("flag is geq patience, best_val_loss_ is: ", best_val_loss) 
            #         stopping_index = True
            #         break
                  


            # else:    

            ###################
            # Early stopping based on val_cindex
            if best_cindex < val_cindex:
                best_cindex = val_cindex
                flag = 0
                # if i >= 2:
                torch.save(model, os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s_%s.pth' %(discr_net, modal, outcome, category, fold_i)))
                print ("Model saved")
            else:
                flag += 1         
                print ("flag: ", flag)           
                if flag >= patience:                        
                    print ("flag is geq patience, best_val_cindex_ is: ", best_cindex) 
                    stopping_index = True
                    break


            # ## Early stopping based on both loss and cindex
            # if best_val_loss > val_loss:
            #     best_val_loss = val_loss
            #     flag_loss = 0
            #     torch.save(model, os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s_%s.pth' %(discr_net, modal, outcome, category, fold_i)))
            #     print ("Model saved")
            # else:
            #     flag_loss += 1         
            #     print ("flag_loss: ", flag_loss)           
            #     if flag_loss >= patience:                        
            #         print ("flag is geq patience, best_val_loss_ is: ", best_val_loss) 
            

            # # val_cindex
            # if best_cindex < val_cindex:
            #     best_cindex = val_cindex
            #     flag_cindex = 0
            #     # if i >= 2:
            #     torch.save(model, os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s_%s.pth' %(discr_net, modal, outcome, category, fold_i)))
            #     print ("Model saved")
            # else:
            #     flag_cindex += 1         
            #     print ("flag_cindex: ", flag_cindex)           
            #     if flag_cindex >= patience:                        
            #         print ("flag is geq patience, best_val_cindex_ is: ", best_cindex) 

            
            # if (flag_cindex >= patience) and (flag_loss >= patience):
            #     break
                



        else:
            print ("Train_loss - %.4f" %train_loss)
            train_loss_hist.append(train_loss)

        if scheduler is not None:
            scheduler.step()

    if evaluator is not None:
      print ("plot training curves")
      fig = plt.figure()
      plt.plot(train_loss_hist, label='Train loss')
      plt.plot(eval_loss_hist, label='Validation loss')
      plt.xlabel("epochs", fontsize=16)
      plt.title("Training loss %s on RS" %discr_net, fontsize=16)
      plt.legend()
      plt.savefig(os.path.join(figure_dir, 'training_loss_curve_%s_%s_%s_%s_%s.png' %(modal, category, discr_net, outcome, fold_i)))

      fig = plt.figure()
      plt.plot(train_cindex_hist, label='Train C-index')
      plt.plot(eval_cindex_hist, label='Validation C-index')
      plt.xlabel("epochs", fontsize=16)
      plt.title("Training C-index %s on RS" %discr_net, fontsize=16)
      plt.legend()
      plt.savefig(os.path.join(figure_dir, 'training_cindex_curve_%s_%s_%s_%s_%s.png' %(modal, category, discr_net, outcome, fold_i)))

    else:
      print ("plot training curves")
      fig = plt.figure()
      plt.plot(train_loss_hist, label='Train loss')
      plt.xlabel("epochs", fontsize=16)
      plt.title("Training %s on RS" %discr_net, fontsize=16)
      plt.legend()
      plt.savefig(os.path.join(figure_dir, 'training_loss_curve_%s_%s_%s_%_%s.png' %(modal, category, discr_net, outcome, fold_i)))
    

    
class ModelRunner:
    """Base class for calling a model on every batch of data.

    Args:
      model (BaseModel):
        Instance of model to call.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`.
      device (torch.device):
        Optional; Which device to run on.
      hooks (list of Hook):
        Optional; List of hooks to call during execution.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """

    def __init__(
        self,
        model: BaseModel,
        data: DataLoaderWrapper,
        device: Optional[torch.device] = None,
        hooks: Optional[Sequence[Hook]] = None,
        progressbar: bool = True,
    ) -> None:
        if device is not None:
            model = model.to(device)

        self.model = model
        self.data = data
        self.device = device
        self.hooks = hooks or []
        self.progressbar = progressbar

    def _dispatch(self, func: str, *args) -> None:
        with torch.no_grad():
            for h in self.hooks:
                fn = getattr(h, func)
                fn(*args)

    def _batch_to_device(self, batch: Union[Tensor, Sequence[Tensor]]) -> Dict[str, Tensor]:
        if not isinstance(batch, (list, tuple)):
            batch = tuple(batch,)

        assert len(self.data.output_names) == len(
            batch
        ), "output_names suggests {:d} tensors, but only found {:d} outputs".format(
            len(self.data.output_names), len(batch)
        )

        batch = dict(zip(self.data.output_names, batch))

        if self.device is not None:
            for k, v in batch.items():
                batch[k] = v.to(self.device)

        return batch

    def _set_model_state(self) -> None:
        pass

    # def run(self) -> None:
    def run(self):
        """Execute model for every batch."""
        self._set_model_state()
        self._dispatch("on_begin_epoch")
        running_loss = 0.
        last_loss = 0.
        pbar = tqdm(self.data, total=len(self.data), disable=not self.progressbar)

        output_list = []
        event_list = []
        time_list = []

        for i, batch in enumerate(pbar):
            batch = self._batch_to_device(batch)
            num_batch = len(self.data)
            # print ("number of batch: ", len(self.data))
            # print ("batch", batch)
            self._dispatch("before_step", batch)
            outputs = self._step(batch)
            # print ("outputs", outputs)
            self._dispatch("after_step", outputs)

            batch_output = outputs['logits'].detach().cpu().numpy().squeeze(1)
            batch_event = batch['event'].detach().cpu().numpy().squeeze(1)
            batch_time = batch['time'].detach().cpu().numpy()

            output_list.append(batch_output)
            event_list.append(batch_event) 
            time_list.append(batch_time) 

            # Gather data and report
            # loss = outputs['total_loss'].detach().cpu().numpy()
            loss = outputs['total_loss']
            running_loss += loss.item()
            # if i % 1000 == 999:
            #     last_loss = running_loss / 100 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     running_loss = 0.

            # if i % 10 == 9:
            #     last_loss = running_loss / 10 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     running_loss = 0.

            if i % num_batch == num_batch-1:
                last_loss = running_loss / num_batch # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        self._dispatch("on_end_epoch")

        ## Concatenate outputs and calculate cindex
        output_epoch = np.concatenate(output_list)
        event_epoch = np.concatenate(event_list).astype(bool)
        time_epoch = np.concatenate(time_list)
        cindex_harrell = concordance_index_censored(event_epoch, time_epoch, output_epoch)

        # return outputs, self.model        
        return last_loss, cindex_harrell[0]
        # return running_loss, cindex_harrell[0]

    def _get_model_inputs_from_batch(self, batch: Dict[str, Tensor]) -> Sequence[Tensor]:
        assert len(batch) >= len(self.model.input_names), "model expects {:d} inputs, but batch has only {:d}".format(
            len(self.model.input_names), len(batch)
        )

        in_batch = tuple(batch[k] for k in self.model.input_names)
        return in_batch

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        in_batch = self._get_model_inputs_from_batch(batch)
        out_tensors = self.model(*in_batch)
        assert len(out_tensors) == len(
            self.model.output_names
        ), "output_names suggests {:d} tensors, but only found {:d} outputs".format(
            len(self.model.output_names), len(out_tensors),
        )

        return out_tensors


class ModelEvaluator(ModelRunner):
    """Execute a model on every batch of data in evaluation mode.

    Args:
      model (BaseModel):
        Instance of model to call.
      loss (BaseModel):
        Instance of loss to compute.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      device (torch.device):
        Optional; Which device to run on.
      hooks (list of Hook):
        Optional; List of hooks to call during execution.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """

    def __init__(
        self,
        model: BaseModel,
        loss: BaseModel,
        data: DataLoaderWrapper,
        device: Optional[torch.device] = None,
        hooks: Optional[Sequence[Hook]] = None,
        progressbar: bool = True,
    ) -> None:
        super().__init__(
            model=model, data=data, device=device, hooks=hooks, progressbar=progressbar,
        )
        all_names = list(chain(model.input_names, model.output_names, loss.output_names))
        check_is_unique(all_names)

        if "total_loss" in all_names:
            raise ValueError("total_loss cannot be used as input or output name")

        model_loss_intersect = set(model.output_names).intersection(set(loss.input_names))
        if len(model_loss_intersect) == 0:
            raise ValueError("model outputs and loss inputs do not agree")

        model_data_intersect = set(model.input_names).intersection(set(data.output_names))
        if len(model_data_intersect) == 0:
            raise ValueError("model inputs and data loader outputs do not agree")

        self.loss = loss

    def _set_model_state(self):
        self.model = self.model.eval()

    def _step_with_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = super()._step(batch)

        batch.update(outputs)
        loss_inputs = [batch[key] for key in self.loss.input_names]

        losses = self.loss(*loss_inputs)
        
        # print ("batch[event]", batch['event'])
        # print ("losses", losses)

        outputs["total_loss"] = sum(losses.values())
        outputs.update(losses)
        # print ("outputs", outputs)
        # print ("outputs",  outputs["total_loss"] )
        return outputs

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        with torch.no_grad():
            return self._step_with_loss(batch)


class ModelTrainer(ModelEvaluator):
    """Execute a model on every batch of data in train mode, compute the gradients, and update the weights.

    Args:
      model (BaseModel):
        Instance of model to call.
      loss (BaseModel):
        Instance of loss to compute.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      optimizer (Optimizer):
        Instance of Optimizer to use.
      device (torch.device):
        Optional; Which device to run on.
      hooks (list of Hook):
        Optional; List of hooks to call during execution.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """

    def __init__(
        self,
        model: BaseModel,
        loss: BaseModel,
        data: DataLoaderWrapper,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
        hooks: Optional[Sequence[Hook]] = None,
        progressbar: bool = True,
    ) -> None:
        super().__init__(
            model=model, loss=loss, data=data, device=device, hooks=hooks, progressbar=progressbar,
        )

        self.optimizer = optimizer

    def _set_model_state(self):
        self.model = self.model.train()

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.optimizer.zero_grad()
        outputs = super()._step_with_loss(batch)

        total_loss = outputs["total_loss"]

        total_loss.backward()
        
        self.optimizer.step()

        return outputs
