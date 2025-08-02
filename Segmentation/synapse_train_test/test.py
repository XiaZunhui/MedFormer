import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from MAXFormer import MAXFormer
from datasets.dataset_synapse import Synapse_dataset
from Mynet.UperHead_medformer import UPerHead
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="/home/cqut/Data/medical_seg_data/Synapse_zheng",
    help="root dir for validation volume data",
)  # for acdc volume_path=root_dir
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--output_dir", type=str, default="./save_models", help="output dir")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
# parser.add_argument("--is_savenii", action="store_true",default=True,help="whether to save results during inference")
parser.add_argument("--is_savenii", action="store_true",default=False,help="whether to save results during inference")

# parser.add_argument("--test_save_dir", type=str, default="../predictions", help="saving prediction as nii!")
parser.add_argument("--test_save_dir", type=str, help="saving prediction as nii!")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs="+",
)
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
# config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    # 初始化存储所有metric结果的列表
    all_metrics = []

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        all_metrics.append(metric_i)  # 保存每个batch的结果

        logging.info(
            "idx %d case %s mean_dice %f mean_hd95 %f"
            % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
        )

    # 转换为numpy数组便于计算
    all_metrics = np.array(all_metrics)  # shape: (num_samples, num_classes, 2)

    # 计算各类别指标的平均值和标准差
    class_dice = all_metrics[:, :, 0]  # 提取所有样本的dice系数
    class_hd95 = all_metrics[:, :, 1]  # 提取所有样本的hd95距离

    mean_dice = np.mean(class_dice, axis=0)
    std_dice = np.std(class_dice, axis=0)
    mean_hd95 = np.mean(class_hd95, axis=0)
    std_hd95 = np.std(class_hd95, axis=0)

    # 输出每个类别的详细统计信息
    for i in range(1, args.num_classes):
        logging.info("Class %d - Dice: %.4f±%.4f | HD95: %.4f±%.4f" %
                     (i, mean_dice[i - 1], std_dice[i - 1], mean_hd95[i - 1], std_hd95[i - 1]))

    # 计算整体性能
    overall_dice = np.mean(class_dice)
    overall_hd95 = np.mean(class_hd95)
    overall_dice_std = np.std(class_dice)
    overall_hd95_std = np.std(class_hd95)

    logging.info("Overall Performance:")
    logging.info("Mean Dice: %.4f±%.4f | Mean HD95: %.4f±%.4f" %
                 (overall_dice, overall_dice_std, overall_hd95, overall_hd95_std))

    return "Testing Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "z_spacing": 1,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    # net = UPerHead(in_channels=[32, 64, 128, 256], channels=64, image_size=(224, 224)).cuda(0)
    # net = UPerHead(in_channels=[64, 128, 256, 512], channels=64, image_size=(224, 224)).cuda(0)
    # net = UPerHead(in_channels=[96, 192, 384, 768], channels=128, image_size=(224, 224)).cuda(0)
    net = MAXFormer(num_classes=9).cuda(0)
    # snapshot = os.path.join(args.output_dir, "best_model.pth")
    # if not os.path.exists(snapshot):
    #     snapshot = snapshot.replace("best_model", "synapse_epoch_299")
    snapshot = '/home/xiazunhui/project/MedFormer/Segmentation/synapse_train_test/synapse_8366.pth'


    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)
    snapshot_name = snapshot.split("/")[-1]

    log_folder = "./test_log/test_log_"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    print(args.is_savenii)
    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
