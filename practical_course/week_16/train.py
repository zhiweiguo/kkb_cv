from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.regularization import label_smoothing
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--start_epoch", type=int, default=0, help="number of atart epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--data_aug_mode", type=int, default=1, help="0:不做处理;1:水平翻转2:GridMask;3:Mosaic;4:GridMask+Mosaic")
    parser.add_argument("--regu_mode", type=int, default=0, help="0:不使用正则化操作;1:使用label smooth;2:使用dropblock;3:两者均使用")
    parser.add_argument("--active_mode", type=int, default=0, help="0:原始激活方式;1:Swish;2:Mish")
    parser.add_argument("--conv_mode", type=int, default=0, help="0:标准卷积;1:depthwise卷积")
    opt = parser.parse_args()
    import pdb
    pdb.set_trace()
    print(opt)

    #import pdb
    #pdb.set_trace()

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def, regu_mode=opt.regu_mode, active_mode=opt.active_mode, conv_mode=opt.conv_mode).to(device)       # default="config/yolov3.cfg"
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
        print("成功加载预训练模型参数:{}".format(opt.pretrained_weights))

    # Get dataloader
    dataset = ListDataset(train_path, img_size=opt.img_size, augment=opt.data_aug_mode, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        #num_workers=opt.n_cpu,
        num_workers=0,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    init_lr = 0.001/(10**(opt.start_epoch//2))
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # 增加学习率衰减方式
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    AP_best = 0
    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            #import pdb
            #pdb.set_trace()
            # label smooth
            if opt.regu_mode == 1 or opt.regu_mode == 3:
                targets[:, 1] = label_smoothing(targets[:, 1], epsilon=0.1)            
            imgs = imgs.to(device)
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)        # 对应darknet中的forward
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0 and AP_best < AP.mean():
            #torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
            torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_{}_{:.4f}.pth".format(epoch, AP.mean()))
            AP_best = AP.mean()
