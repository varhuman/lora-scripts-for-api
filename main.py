import argparse
import os
import shutil
import subprocess
import sys
import webbrowser

import uvicorn
from typing import List

# from mikazuki.app import app

parser = argparse.ArgumentParser(description="GUI for stable diffusion training")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=28000, help="Port to run the server on")
parser.add_argument("--tensorboard-host", type=str, default="127.0.0.1", help="Port to run the tensorboard")
parser.add_argument("--tensorboard-port", type=int, default=6007, help="Port to run the tensorboard")
parser.add_argument("--dev", action="store_true")


def find_windows_git():
    possible_paths = ["git\\bin\\git.exe", "git\\cmd\\git.exe", "Git\\mingw64\\libexec\\git-core\\git.exe"]
    for path in possible_paths:
        if os.path.exists(path):
            return path


def prepare_frontend():
    if not os.path.exists("./frontend/dist"):
        print("Frontend not found, try clone...")
        print("Checking git installation...")
        if not shutil.which("git"):
            if sys.platform == "win32":
                git_path = find_windows_git()

                if git_path is not None:
                    print(f"Git not found, but found git in {git_path}, add it to PATH")
                    os.environ["PATH"] += os.pathsep + os.path.dirname(git_path)
                    return
            else:
                print("Git not found, please install git first")
                sys.exit(1)
        subprocess.run(["git", "submodule", "init"])
        subprocess.run(["git", "submodule", "update"])


def remove_warnings():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if sys.platform == "win32":
        # disable triton on windows
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


def run_tensorboard():
    print("Starting tensorboard...")
    subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", "logs", "--host", args.tensorboard_host, "--port", str(args.tensorboard_port)])


def check_dirs(dirs: List):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def test_find_lora():
    import mikazuki.utils as utils
    logging_dir = "logs/20230725023307"
    tensorboard_log =  utils.find_event_file(logging_dir)
    if tensorboard_log is None:
        print("虽然训练成功，但是找不到tensorboard日志文件！")
    else:
        print("tensorboard_log日志地址在",tensorboard_log)
        model_dir = utils.find_best_model("haha1", "./output", tensorboard_log)
        if model_dir is None:
            print("虽然训练成功，但是找不到最优模型！")
        else:
            print("最好的模型路径是：", model_dir)

def test_tensorboard():
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator("logs/20230725023307/network_train/events.out.tfevents.1690223607.tvpn.63084.0", size_guidance={ # see below regarding this argument
         event_accumulator.COMPRESSED_HISTOGRAMS: 500,
         event_accumulator.IMAGES: 4,
         event_accumulator.AUDIO: 4,
         event_accumulator.SCALARS: 0,
         event_accumulator.HISTOGRAMS: 1,
     })
    ea.Reload()
    #get loss
    print(ea.scalars.Keys())
    # print(ea.scalars.Items('loss/average'))
    # print(ea.scalars.Items('loss/epoch'))
    # print(ea.scalars.Items('loss/epoch')[0][2])
    print(len(ea.scalars.Items('loss/epoch')))
    # print(ea.scalars.Items('loss/epoch'))
    # print(ea.scalars.Items('loss')[0][2])
    # #get lr
    # print(ea.scalars.Keys())
    # print(ea.scalars.Items('lr'))
    # print(ea.scalars.Items('lr')[0][2])
    # #get psnr
    # print(ea.scalars.Keys())
    # print(ea.scalars.Items('psnr'))
    # print(ea.scalars.Items('psnr')[0][2])
    # #get ssim
    # print(ea.scalars.Keys())
    # print(ea.scalars.Items('ssim'))
    # print(ea.scalars.Items('ssim')[0][2])
    # #get val_loss
    # print(ea.scalars.Keys())
    # print(ea.scalars.Items('val_loss'))
    # print(ea.scalars.Items('val_loss')[0][2])
    # #get loss average
    # print(ea.scalars.Keys())
    # print(ea.scalars.Items('loss_average'))
    # print(ea.scalars.Items('loss_average')[0][2])

if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    remove_warnings()
    # prepare_frontend()
    # run_tensorboard()
#     import os
#     import subprocess

    # 获取当前conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not inside a Conda environment')
    print(f'Current Conda environment: {conda_env}')

#     # 获取已安装的pip包
#     process = subprocess.run(["pip", "freeze"], text=True, capture_output=True)
#     pip_packages = process.stdout.splitlines()
#     print('Installed pip packages:')
#     for package in pip_packages:
#         print(package)

    import tensorboard as tb
    print("TensorBoard version: ", tb.__version__)
    # test_tensorboard()
    # test_find_lora()
    print(f"Server started at http://{args.host}:{args.port}")
    uvicorn.run("mikazuki.app_self:app", host=args.host, port=args.port, log_level="error")
