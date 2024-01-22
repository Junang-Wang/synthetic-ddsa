import argparse
import datetime
import gc
import json
import os
import time

import imgaug as ia
import matplotlib
import numpy as np
import torch

# multiGPU
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import utils.auxiliaries_sim as aux
from models.Unet import UNet, init_weights

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
matplotlib.use("Agg")


def main():
    """---------------------------    Setup parser    ---------------------------"""
    parser = argparse.ArgumentParser()

    # -----------------------------   General Setup   ------------------------------#
    parser.add_argument(
        "--savepath",
        default=os.path.join(os.getcwd(), "results_new"),
        help="Where to store the results.",
    )
    parser.add_argument(
        "--datafolder",
        default=os.path.join(os.getcwd(), "data"),
        help="Where the rawdata is located.",
    )
    parser.add_argument(
        "--save_ram",
        action="store_true",
        help='Use pre-saved tmp folder instead. Use "save_tmp.py".',
    )
    parser.add_argument(
        "--use_combine_patches",
        action="store_true",
        help="Use pre-saved tmp folder instead.",
    )
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument(
        "--sim_vessel", action="store_true", help="Use simulated vessels for training"
    )
    parser.add_argument("--m_cuda", action="store_true", help="Enables multiple GPUs")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=0,
        help="Cuda device to use (default = 0)",
    )

    # --------------------------   Training Parameters   ---------------------------#
    parser.add_argument(
        "--nepochs", type=int, default=75, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate, default=0.0001"
    )
    parser.add_argument(
        "--b1", type=float, default=0.9, help="beta1 parameter of Adam."
    )
    parser.add_argument(
        "--b2", type=float, default=0.999, help="beta2 parameter of Adam."
    )
    parser.add_argument("--criterion", default="L1", help="L1 or L2 loss")
    parser.add_argument("--mbs", type=int, default=32, help="Mini-batch size")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers to use for preprocessing.",
    )

    # ----------------------------   Data Parameters   -----------------------------#
    parser.add_argument(
        "--ntrain", type=int, default=36800, help="Number of training samples"
    )
    parser.add_argument(
        "--nval", type=int, default=4800, help="Number of validation samples"
    )
    parser.add_argument(
        "--sample_strategy",
        default="dataset",
        help='Sample strategy for the data. If "dataset", sample \
                              uniform w.r.t to datasets. If "uniform" sample \
                              uniform over all datasets (not recommended).',
    )
    parser.add_argument(
        "--add_noise", action="store_true", help="add poisson noise to CT projections"
    )
    parser.add_argument("--augment", action="store_true", help="Augment the data.")
    parser.add_argument(
        "--pBlur", type=float, default=0.8, help="Probability for blur."
    )
    parser.add_argument(
        "--pAffine",
        type=float,
        default=0.8,
        help="Probability for affine transformations.",
    )
    parser.add_argument(
        "--pMultiply",
        type=float,
        default=0.8,
        help="Probability for multiplicative augmentation.",
    )
    parser.add_argument(
        "--pContrast",
        type=float,
        default=0.8,
        help="Probability for contrast augmentations.",
    )

    parser.add_argument(
        "--init_patch",
        type=int,
        default=512,
        help="Size of patch before data augmentation.",
    )
    parser.add_argument(
        "--final_patch",
        type=int,
        default=384,
        help="Size of patch after data augmentation.",
    )
    parser.add_argument(
        "--normalization",
        default="global",
        help="Data normalization. Global or local (not recommended).",
    )

    # ---------------------------   Network Parameters   ---------------------------#
    parser.add_argument("--init_type", default="kaiming", help="Weight initialization.")
    parser.add_argument("--upmode", default="conv", help="Upsample mode")
    parser.add_argument("--downmode", default="sample", help="Downsample mode")
    parser.add_argument(
        "--batchnorm", action="store_true", help="Use Batch-Normalization"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Use dropout with provided probability.",
    )
    parser.add_argument(
        "--use_trained_model",
        action="store_true",
        help="Use a trained model for model initialization.",
    )
    parser.add_argument(
        "--pretrained_model",
        default="DeepDSA_2023_02_27__041141",
        help="pretrained model name",
    )

    # ------------------------------   Random Seeds   ------------------------------#
    parser.add_argument("--seed", type=int, help="manual seed")
    opt = parser.parse_args()

    # Setup device(s) to use
    if isinstance(opt.devices, list) and len(opt.devices) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.devices])
        opt.devices = list(range(len(opt.devices)))
    elif isinstance(opt.devices, list) and len(opt.devices) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.devices[0])
        opt.devices = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.devices)
        opt.devices = 0

    if opt.m_cuda:
        torch.cuda.set_device(opt.local_rank)
        opt.device = torch.cuda.get_device_name(opt.local_rank)
        # backend initialization
        dist.init_process_group(backend="gloo")
    elif opt.cuda:
        torch.cuda.set_device(0)
        opt.device = torch.cuda.get_device_name(0)
    else:
        opt.device = "cpu"
    # ------------------------------   Setup seeds   -------------------------------#
    if opt.seed is None:
        np.random.seed()
        opt.seed = np.random.randint(1, 10000)
    # opt.seed = 2029
    print("Random Seed: ", opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    ia.seed(opt.seed)
    torch.backends.cudnn.benchmark = True  # speed up
    # ----------------------   Write Hyperparameters file   ------------------------#
    rundate = datetime.datetime.now()
    savetime = "{:02d}_{:02d}_{:02d}__{:02d}{:02d}{:02d}".format(
        rundate.year,
        rundate.month,
        rundate.day,
        rundate.hour,
        rundate.minute,
        rundate.second,
    )
    opt.savepath = os.path.join(opt.savepath, "DeepDSA_" + savetime)
    # if not os.path.exists(opt.savepath):
    #     os.makedirs(opt.savepath)

    # -----------------------------   Setup Logger   -------------------------------#
    if opt.cuda == True or (opt.m_cuda == True and dist.get_rank() == 0):
        if not os.path.exists(opt.savepath):
            os.makedirs(opt.savepath)
        logger = SummaryWriter(log_dir=opt.savepath)

    """----------------------------   Load Dataset    ---------------------------"""
    print("Setup dataloader...")
    start = time.time()
    probs = {
        "Blur": opt.pBlur,
        "Affine": opt.pAffine,
        "Multiply": opt.pMultiply,
        "Contrast": opt.pContrast,
    }

    if opt.save_ram:
        print("use save_ram for train", opt.RamDisk)
        #  normalization values same with original dataset without cropping
        Data_Train = aux.Data_SaveRam(
            opt.ntrain,
            opt.datafolder,
            opt.RamDisk,
            opt.clahe,
            probs=probs,
            mode="train",
            normalization={
                "type": opt.normalization,
                "mean_std": (424.6548767089844, 360.46417236328125),
                "clahe_mean_std": (4608.6997, 11309.425),
                "min_max": None,
            },
            init_patch=opt.init_patch,
            final_patch=opt.final_patch,
            augment=opt.augment,
            sample_strategy=opt.sample_strategy,
            rejection=opt.rejection,
        )
        # after cropping  "mean_std" (561.4381, 341.20062)
    elif opt.sim_vessel:
        print("load synthetic vessels and fluoro mask images")
        opt.vessel_file = ["vessels\\train\\"]
        Data_Train = aux.Data_vessel(
            opt,
            mode="train",
            normalization={
                "type": opt.normalization,
                "mean_std": None,
                "target_mean_std": None,
                "clahe_mean_std": None,
            },
        )
    else:
        print("load synthetic DSA image pairs")
        # Data_Train = aux.Data(opt, mode='train', normalization={
        #                       'type': opt.normalization, 'mean_std': (
        #                                   424.6548767089844, 360.46417236328125), 'clahe_mean_std': (4608.6997, 11309.425)})
        # Data_Train = aux.Data(opt, mode='train', normalization={
        #                       'type': opt.normalization, 'mean_std': None, 'target_mean_std': None})

        # opt.vessel_file = ["vessels\\CT_train_fluid\\"]
        # opt.vessel_file = ["vessels\\CT_train_more\\"]

        # opt.vessel_file = ["vessels\\CT_train\\"]  # non-flow
        # opt.vessel_file = ["vessels\\bolus_chase_t600_train\\"]  # with bolus chase
        # opt.vessel_file = ["vessels\\bolus_chase_t600_train_30_58\\"]  # with bolus chase
        # opt.vessel_file = ["vessels\\flowing_vessels\\train_flow\\"]  # flowing vessels
        opt.vessel_file = [
            "vessels\\flowing_vessels\\30-60_new\\train_flow\\"
        ]  # flowing vessels 30-60
        # opt.vessel_file = ["vessels\\bolus_chase_t600_train\\", 'vessels\\bolus_chase_ts_train\\']
        Data_Train = aux.Data_CTP_sim_equ_mask(
            opt,
            mode="train",
            normalization={
                "type": opt.normalization,
                "mean_std": None,
                "min_max": None,
                "target_mean_std": None,
                "clahe_mean_std": None,
            },
        )  # [432, 223]
    if opt.m_cuda:
        train_sampler = DistributedSampler(Data_Train)
        Dataloader_Train = DataLoader(
            Data_Train,
            batch_size=opt.mbs,
            num_workers=opt.n_workers,
            worker_init_fn=aux.worker_init_fn,
            sampler=train_sampler,
            pin_memory=True,
        )
    else:
        Dataloader_Train = DataLoader(
            Data_Train,
            batch_size=opt.mbs,
            shuffle=True,
            num_workers=opt.n_workers,
            worker_init_fn=aux.worker_init_fn,
            pin_memory=True,
        )
    print(Data_Train.normalization)
    Train_mean_std = (
        Data_Train.normalization["mean_std"]
        if opt.normalization == "global" or opt.normalization == "local"
        else None
    )
    Train_mean_std = (
        Train_mean_std + Data_Train.normalization["clahe_mean_std"]
        if opt.clahe
        else Train_mean_std
    )
    opt.mean_std = (
        [float(x) for x in Train_mean_std]
        if opt.normalization == "global" or opt.normalization == "local"
        else None
    )
    Train_min_max = (
        Data_Train.normalization["min_max"]
        if opt.normalization == "min_max" or opt.normalization == "local_min_max"
        else None
    )
    opt.min_max = (
        [float(x) for x in Train_min_max]
        if opt.normalization == "min_max" or opt.normalization == "local_min_max"
        else None
    )

    opt.vessel_file = ["vessels\\test\\"]
    Data_Val = aux.Data_CTP_sim_equ_mask(
        opt, mode="val", normalization=Data_Train.normalization
    )

    if opt.m_cuda:
        val_sampler = DistributedSampler(Data_Val)
        Dataloader_Val = DataLoader(
            Data_Val,
            batch_size=opt.mbs,
            num_workers=opt.n_workers,
            worker_init_fn=aux.worker_init_fn,
            sampler=val_sampler,
            pin_memory=True,
        )
    else:
        Dataloader_Val = DataLoader(
            Data_Val,
            batch_size=opt.mbs,
            shuffle=True,
            num_workers=opt.n_workers,
            worker_init_fn=aux.worker_init_fn,
            pin_memory=True,
        )
    print("...finished")

    stop = time.time()

    time_elapsed = stop - start
    print(
        "Dataloader complates in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    with open(os.path.join(opt.savepath, "Hyperparameters.txt"), "w") as f:
        json.dump(vars(opt), f)

    # import sys
    # sys.exit(0)

    """-----------------------------------------------------------------------------
    --------------------   Setup network, loss & optimizers   ----------------------
    -----------------------------------------------------------------------------"""
    print("Setup the network and loss...")
    ch = [64, 128, 256, 512, 1024]
    # -----------------------------   Setup network  -------------------------------#
    net = UNet(
        ch=ch,
        downmode=opt.downmode,
        upmode=opt.upmode,
        batchnorm=opt.batchnorm,
        dropout=opt.dropout,
        clahe=opt.clahe,
    )
    if opt.use_trained_model:
        checkpoint = torch.load(
            os.path.join(
                os.getcwd(), "results", opt.pretrained_model, "best_train_net.pt"
            )
        )
        net.load_state_dict(checkpoint["model"])
    else:
        init_weights(net, init_type=opt.init_type)

    if opt.m_cuda:
        net.cuda()
        print("send network to multiple cuda")
        net = DDP(net, device_ids=[opt.local_rank])
    elif opt.cuda:
        print("send network to cuda")
        net.cuda()

    # ------------------------------   Setup loss  ---------------------------------#
    if opt.criterion == "L1":
        pix_crit = nn.L1Loss()
    elif opt.criterion == "L2":
        pix_crit = nn.MSELoss()
    if opt.m_cuda or opt.cuda:
        pix_crit = pix_crit.cuda()
    # ---------------------------   Setup optimizers  ------------------------------#
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=15, verbose=True, threshold=1e-3
    )
    print("...finished")

    """-----------------------------------------------------------------------------
    -----------------------------   Train network    -------------------------------
    -----------------------------------------------------------------------------"""
    print("Start training...")
    start = time.time()
    best_val_loss = np.inf
    best_train_loss = np.inf
    for epoch in range(opt.nepochs):
        np.random.seed(np.random.randint(1, 10000))
        train_loss = 0.0
        val_loss = 0.0
        start_epoch = time.time()

        if opt.m_cuda:
            Dataloader_Train.sampler.set_epoch(epoch)
            Dataloader_Val.sampler.set_epoch(epoch)

        """-------------------------    training    -------------------------"""
        net.train()
        for i_batch, sample_batched in enumerate(Dataloader_Train):
            input, target = Variable(sample_batched["x"]), Variable(sample_batched["y"])

            if opt.m_cuda or opt.cuda:
                input, target = input.cuda(), target.cuda()
            output = net(input)
            pix_loss = pix_crit(output, target)
            train_loss += pix_loss.data.item()

            optimizer.zero_grad()
            pix_loss.backward()
            optimizer.step()

            print(
                "[TRAIN: epoch %2d of %2d | minibatch %3d of %3d | loss %.4f]"
                % (
                    epoch + 1,
                    opt.nepochs,
                    i_batch + 1,
                    len(Dataloader_Train),
                    pix_loss.data.item(),
                )
            )

            if not opt.m_cuda or dist.get_rank() == 0:
                # if epoch == 0:
                # logger.add_graph(net, input)
                patches = {
                    "input": input[:6, 0, :, :].data.cpu().numpy(),
                    "target": target[:6, 0, :, :].data.cpu().numpy(),
                    "output": output[:6, 0, :, :].data.cpu().numpy(),
                }
            del input, target, pix_loss, output
            gc.collect()
            torch.cuda.empty_cache()

        train_loss /= len(Dataloader_Train)

        """-------------------------    validation    -------------------------"""
        net.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(Dataloader_Val):
                input, target = Variable(sample_batched["x"]), Variable(
                    sample_batched["y"]
                )

                if opt.m_cuda or opt.cuda:
                    input, target = input.cuda(), target.cuda()

                output = net(input)
                pix_loss = pix_crit(output, target)
                val_loss += pix_loss.data.item()

                if not opt.m_cuda or dist.get_rank() == 0:
                    if i_batch == 0:
                        val_patches = {
                            "input": input[:6, 0, :, :].data.cpu().numpy(),
                            "target": target[:6, 0, :, :].data.cpu().numpy(),
                            "output": output[:6, 0, :, :].data.cpu().numpy(),
                        }
                del input, target, pix_loss

                gc.collect()
                torch.cuda.empty_cache()
        val_loss /= len(Dataloader_Val)
        print(
            "[Validation: epoch %2d of %2d loss %.4f]"
            % (epoch + 1, opt.nepochs, val_loss)
        )
        scheduler.step(val_loss)

        """--------------------------    Save Logs    ---------------------------"""
        if not opt.m_cuda or dist.get_rank() == 0:
            stop = time.time()

            time_elapsed = stop - start_epoch
            print(
                "This epoch completes in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

            aux.write_log(
                os.path.join(opt.savepath, "Log.csv"),
                epoch,
                stop - start,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
            )
            aux.make_learning_curves_fig(os.path.join(opt.savepath, "Log.csv"))

            """------------------------    Save Networks    ------------------------"""
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print("Apply to train stacks")
                train_log_imgs = aux.apply_to_stacks(
                    Data_Train, [0], net, epoch, opt, Data_Train.normalization
                )

                if opt.m_cuda:
                    state_train = {
                        "model": net.module.state_dict(),
                        "normalization": Data_Train.normalization,
                    }
                else:
                    state_train = {
                        "model": net.state_dict(),
                        "normalization": Data_Train.normalization,
                    }
                torch.save(state_train, os.path.join(opt.savepath, "best_train_net.pt"))

                logger.add_images(
                    "train applied",
                    aux.min_max_norm(
                        np.expand_dims(np.stack(train_log_imgs["applied"], 0), 1)
                    ),
                    epoch + 1,
                )
                if epoch == 0:
                    logger.add_images(
                        "train target",
                        aux.min_max_norm(
                            np.expand_dims(np.stack(train_log_imgs["target"], 0), 1)
                        ),
                        epoch + 1,
                    )

            if val_loss < best_val_loss:
                if opt.m_cuda:
                    state_val = {
                        "model": net.module.state_dict(),
                        "normalization": Data_Train.normalization,
                    }
                else:
                    state_val = {
                        "model": net.state_dict(),
                        "normalization": Data_Train.normalization,
                    }
                torch.save(state_val, os.path.join(opt.savepath, "best_val_net.pt"))
            """----------------------    Tensorboard Logs    --------------------------"""
            # Tensorboard Logging
            losses = {"train loss": train_loss, "val loss": val_loss}
            for tag, value in losses.items():
                logger.add_scalar(tag, value, epoch + 1)

            logger.add_images(
                "patch input",
                aux.min_max_norm(np.expand_dims(patches["input"], 1)),
                epoch + 1,
            )
            logger.add_images(
                "patch target",
                aux.min_max_norm(np.expand_dims(patches["target"], 1)),
                epoch + 1,
            )
            logger.add_images(
                "patch output",
                aux.min_max_norm(np.expand_dims(patches["output"], 1)),
                epoch + 1,
            )

            logger.add_images(
                "val patch input",
                aux.min_max_norm(np.expand_dims(val_patches["input"], 1)),
                epoch + 1,
            )
            logger.add_images(
                "val patch target",
                aux.min_max_norm(np.expand_dims(val_patches["target"], 1)),
                epoch + 1,
            )
            logger.add_images(
                "val patch output",
                aux.min_max_norm(np.expand_dims(val_patches["output"], 1)),
                epoch + 1,
            )


if __name__ == "__main__":
    main()
