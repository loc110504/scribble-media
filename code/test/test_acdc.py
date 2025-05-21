import sys
import os
# __file__ = .../DMSPS/code/test/test_acdc.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR = .../DMSPS/code
sys.path.insert(0, BASE_DIR)
import argparse
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import time
import logging

from networks.net_factory import net_factory
from utils import calculate_metric_percase, logInference

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='../../data/ACDC2017/ACDC_for2D',
                    help='Root folder chứa các thư mục *_volumes và file .txt')
parser.add_argument('--testData', type=str,
                    default='test.txt',
                    help='File .txt liệt kê danh sách TestSet_volumes/*.h5')
parser.add_argument('--savedir', type=str,
                    default='TsResult',
                    help='Tên thư mục con kết quả (TsResult, ValResult, TrResult)')
parser.add_argument('--model', type=str,
                    default='unet_cct',
                    help='Model name: unet, unet_cct, ...')
parser.add_argument('--exp', type=str,
                    default='A_weakly_SPS_2d',
                    help='Experiment name')
parser.add_argument('--fold', type=str,
                    default='stage1',
                    help='Fold name (ví dụ stage1)')
parser.add_argument('--num_classes', type=int,
                    default=4, help='Số lớp phân đoạn')
parser.add_argument('--patch_size', type=list,
                    default=[256, 256],
                    help='Kích thước input [H, W]')
parser.add_argument('--tt_num', type=int,
                    default=1, help='Chọn branch 1 hay 2 nếu dual-branch')

def test_single_volume_2d(case_path, net, out_dir, FLAGS):
    case_name = os.path.splitext(os.path.basename(case_path))[0]
    logging.info(f"Testing case: {case_name}")

    # --- 1) Đọc HDF5 ---
    with h5py.File(case_path, 'r') as f:
        image = f['image'][:]    # shape (Z, H, W), dtype=float64
        label = f['label'][:]    # shape (Z, H, W), dtype=uint8

    # chuẩn bị mảng prediction
    prediction = np.zeros_like(label, dtype=np.uint8)

    # --- 2) Inference từng slice ---
    net.eval()
    for z in range(image.shape[0]):
        sl = image[z, :, :].astype(np.float32)
        h, w = sl.shape

        # resize về patch_size
        sl_resized = zoom(sl,
                          (FLAGS.patch_size[0] / h,
                           FLAGS.patch_size[1] / w),
                          order=0)

        inp = torch.from_numpy(sl_resized)\
                   .unsqueeze(0).unsqueeze(0)\
                   .float().cuda()

        with torch.no_grad():
            if FLAGS.model in ("unet_cct", "unet_2dual"):
                # chọn branch
                if FLAGS.tt_num == 2:
                    _, out_main = net(inp)
                else:
                    out_main, _ = net(inp)
            else:
                out_main = net(inp)

            # softmax → argmax
            prob = torch.softmax(out_main, dim=1)
            pred_cls = torch.argmax(prob, dim=1).squeeze(0)
            pred_np = pred_cls.cpu().numpy().astype(np.uint8)

        # resize ngược về độ phân giải gốc
        pred_orig = zoom(pred_np,
                         (h / FLAGS.patch_size[0],
                          w / FLAGS.patch_size[1]),
                         order=0).astype(np.uint8)

        prediction[z] = pred_orig

    # --- 3) Tính metric (giả spacing) ---
    spacing = (1.0, 1.0, 1.0)
    metrics = []
    for i in range(1, FLAGS.num_classes):
        m = calculate_metric_percase(
            prediction == i,
            label == i,
            spacing
        )
        metrics.append(m)

    # --- 4) Lưu kết quả dưới dạng NIfTI ---
    # Giả lập metadata
    def save_nii(arr, filename):
        itk = sitk.GetImageFromArray(arr.astype(np.float32))
        itk.SetSpacing(spacing)
        sitk.WriteImage(itk, filename)

    save_nii(prediction, os.path.join(out_dir, f"{case_name}_pred.nii.gz"))
    save_nii(label,      os.path.join(out_dir, f"{case_name}_gt.nii.gz"))
    save_nii(image,      os.path.join(out_dir, f"{case_name}_img.nii.gz"))

    return metrics


def Inference(FLAGS, out_dir):
    # đọc danh sách .h5
    txt_path = os.path.join(FLAGS.data_root_path, FLAGS.testData)
    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    h5_list = [os.path.join(FLAGS.data_root_path, p) for p in lines]
    logging.info(f"Total cases to test: {len(h5_list)}")

    # load model checkpoint
    snap = os.path.join(
        "../../model",
        f"{FLAGS.data_type}_{FLAGS.data_name}",
        f"{FLAGS.exp}_{FLAGS.model}_{FLAGS.fold}",
        f"{FLAGS.model}_best_model.pth"
    )
    net = net_factory(net_type=FLAGS.model,
                      in_chns=1,
                      class_num=FLAGS.num_classes)
    net.load_state_dict(torch.load(snap))
    net.cuda().eval()
    logging.info(f"Loaded checkpoint: {snap}")

    all_metrics = []
    for case_h5 in tqdm(h5_list):
        m = test_single_volume_2d(case_h5, net, out_dir, FLAGS)
        all_metrics.append(m)

    # log summary
    logInference(all_metrics)


if __name__ == "__main__":
    start = time.time()
    FLAGS = parser.parse_args()

    # tạo thư mục lưu kết quả
    out_dir = os.path.join(
        "../../result",
        f"{FLAGS.data_type}_{FLAGS.data_name}",
        f"{FLAGS.exp}_{FLAGS.model}_{FLAGS.savedir}_{FLAGS.tt_num}"
    )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # thiết lập logging
    logger = logging.getLogger()
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(out_dir, "test_info.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    logging.info("Flags:")
    logging.info(FLAGS)

    # run
    Inference(FLAGS, out_dir)

    elapsed = time.time() - start
    logging.info(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
