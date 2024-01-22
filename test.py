import argparse
import json
import os
import time

import matplotlib
import torch

import utils.auxiliaries as aux
from models.Unet import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use("Agg")


def main():
    for m in range(len(model_paths)):
        model_path = model_paths[m]
        print(model_path)
        # ------------------------   load parameters   ----------------------------#
        Hyperparameters = json.load(
            open(
                os.path.join(os.getcwd(), "results", model_path, "Hyperparameters.txt"),
                "r",
            )
        )
        opt = argparse.Namespace(**Hyperparameters)

        opt.datafolder = os.path.join(os.getcwd(), "data")

        if hasattr(opt, "m_cuda") and opt.m_cuda or opt.cuda:
            torch.cuda.set_device(0)
            opt.device = torch.cuda.get_device_name(0)
        else:
            opt.device = "cpu"

        torch.backends.cudnn.benchmark = True

        """----------------------------   set network    ---------------------------"""

        start = time.time()
        ch = [64, 128, 256, 512, 1024]
        net = UNet(
            ch=ch,
            downmode=opt.downmode,
            upmode=opt.upmode,
            batchnorm=opt.batchnorm,
            dropout=opt.dropout,
            clahe=opt.clahe,
        )

        if hasattr(opt, "m_cuda") and opt.m_cuda or opt.cuda:
            print("send network to cuda")
            net.cuda()

        # ---------------------------torch.save(net.state_dict())----------#
        # net.load_state_dict(torch.load(os.path.join(opt.savepath,model_path)))
        # net.eval()

        # -----------------normalization saved----------------------#
        checkpoint = torch.load(
            os.path.join(os.getcwd(), "results", model_path, model_name)
        )
        net.load_state_dict(checkpoint["model"])
        normalization = checkpoint["normalization"]
        print(normalization)
        net.eval()

        """-------------------------    validation    -------------------------"""
        opt.savepath = os.path.join(
            os.getcwd(), results_folder, model_path
        )  # changed for restoring valuation results
        if not os.path.exists(opt.savepath):
            os.makedirs(opt.savepath)

        for num in range(len(eval_names)):
            video_size = video_sizes[num]
            eval_name = eval_names[num]
            print("Apply to val %d in %d" % (num + 1, len(eval_names)))
            # print(normalization['type'])
            if normalization["type"] == "local":
                normalization["type"] = "global"
            aux.apply_to_raw(
                eval_name,
                model_name[5:8],
                video_size[0],
                video_size[1],
                video_size[2],
                net,
                0,
                opt,
                normalization,
                {"applied": [], "target": []},
                self_norm=self_norm,
            )

        """--------------------------    Save Logs    ---------------------------"""
        stop = time.time()
        time_elapsed = stop - start
        print(
            "Valuation complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )


if __name__ == "__main__":
    # ------------------  model path   -------------#
    model_paths = ["DeepDSA_2023_04_10__065130"]
    model_name = "best_val_net.pt"
    results_folder = "results_val"
    self_norm = True  # Normalize test data with the mean and standard deviation of the test data itself

    # # -------------validation---------------x y z
    eval_names = ["Fluoro_1_1024x1024x127.raw"]
    video_sizes = [[1024, 1024, 127]]
    main()
