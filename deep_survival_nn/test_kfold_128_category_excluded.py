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
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
import os
from daft.cli import HeterogeneousModelFactory, create_parser
from daft.training.hooks import CheckpointSaver, TensorBoardLogger
from daft.training.train_and_eval import train_and_evaluate
# from daft.evaluate import get_data_with_logits
import numpy as np
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored
import scipy.stats as st

from scipy.special import logit, expit

import random 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
daft_dir = os.path.join(current_dir, 'daft')
trained_model_dir = os.path.join(daft_dir, 'trained_model')

k_of_fold = 10

# print ("checkpoint1")

def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI

def main(args=None):
    args = create_parser().parse_args(args=args)
    # print ("checkpoint2")
    discr_net = args.discriminator_net
    outcome = args.outcome

    print (discr_net)
    print (outcome)


    test_cindex_list = []

    current_dir = os.path.dirname(os.path.abspath(__file__)) #/trinity/home/hmo/hmo/ergo_ct/DAFT/
    hdf_dirname = "hdf_cac_128"
    hdf_dir = os.path.join(current_dir, hdf_dirname)

    category = args.category
    modal = args.imgmodal

    print ("category", category)


    for i in range(k_of_fold):
        print ("k=", i)
        args.train_data = os.path.join(hdf_dir, "ergo_train_CHD_128_%s_excluded_%s.hdf5" %(category, i)  )
        args.val_data = os.path.join(hdf_dir, "ergo_val_CHD_128_%s_excluded_%s.hdf5" %(category, i)  )
        args.test_data = os.path.join(hdf_dir, "ergo_test_CHD_128_%s_excluded_%s.hdf5" %(category, i)  )


        factory = HeterogeneousModelFactory(args)

        experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()
        factory.write_args(experiment_dir / "experiment_args.json")
        train_loader, valid_loader, test_loader = factory.get_data()

        # dev = torch.device("cuda")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        ## Load saved model 
        # if torch.cuda.is_available():
        #     model = torch.load(os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s.pth' %(discr_net, modal, outcome, i)))
        # else:
        #     model = torch.load(os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s.pth' %(discr_net, modal, outcome, i)), map_location=torch.device('cpu'))

        # if torch.cuda.is_available():
        #     model = torch.load(os.path.join(trained_model_dir, '%s_trained_network_%s_%s.pth' %(discr_net, outcome, i)))
        # else:
        #     model = torch.load(os.path.join(trained_model_dir, '%s_trained_network_%s_%s.pth' %(discr_net, outcome, i)), map_location=torch.device('cpu'))


        ## Load saved model 
        if torch.cuda.is_available():
            model = torch.load(os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s_%s.pth' %(discr_net, modal, outcome, category, i)))
        else:
            model = torch.load(os.path.join(trained_model_dir, '%s_trained_network_%s_%s_%s_%s.pth' %(discr_net, modal, outcome, category, i)), map_location=torch.device('cpu'))


        test_out = {
            "event": [],
            "time": [],
            "logits": [],
        }

        ## Evaluation mode
        model = model.to(dev).eval()

        with torch.no_grad():
            for image, tabular, event, time, _riskset in test_loader:
                test_out["event"].append(event.squeeze(1))
                test_out["time"].append(time)

                # print ("image.shape", image.shape)
                # print ("tabular.shape", tabular.shape)
                # print ("event.shape", event.shape)
                # print ("time.shape", time.shape)

                outputs = model(image.to(dev), tabular.to(dev))
                logits = outputs["logits"].detach().cpu()
                test_out["logits"].append(logits.squeeze(1))

        for k, v in test_out.items():
            test_out[k] = torch.cat(v).numpy()

        test_out["y"] = np.fromiter(zip(test_out.pop("event"), test_out.pop("time")), dtype=[("event", bool), ("time", float)])

        test_y = test_out["y"]
        logits = test_out["logits"]

        cindex_harrell = concordance_index_censored(test_y["event"], test_y["time"], logits)
        print ("cindex_harrell", cindex_harrell)
        print ("concordance/cindex", cindex_harrell[0])

        test_cindex_list.append(cindex_harrell[0])

    test_mean = np.mean(test_cindex_list)
    test_std = np.std(test_cindex_list)
    print ("test_cindex_list", test_cindex_list)
    print ("test_mean +- std %s +- %s" %(round(test_mean, 4), round(test_std, 4)))

    ci = compute_confidence(np.array(test_cindex_list), 1600, 400, alpha=0.95)
    print ("ci", ci)

    metrics = test_cindex_list
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics)-1, loc=mean, scale=st.sem(metrics))

    print ('mean', mean)
    print ('confidence_interval', conf_interval)


if __name__ == "__main__":
    main()
