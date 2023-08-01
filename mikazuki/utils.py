import os
import sys
import glob
import subprocess
import importlib.util
from typing import Optional
import glob

python_bin = sys.executable

def find_event_file(directory):
    # 遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否以 'event' 开头
            if file.startswith('event'):
                # return full path
                return os.path.join(root, file)
    return None

def change_model_name(model_path: str, old_name: str, new_name: str) -> bool:
    print(f"model_path: {model_path} old_name {old_name}  new_name {new_name}")
    if not os.path.exists(model_path):
        return False
    # model_old_path is file path
    model_old_path = os.path.join(model_path, old_name)
    if os.path.isfile(model_old_path):
        model_new_path = os.path.join(model_path, new_name)
        os.rename(model_old_path, model_new_path)
        return True
    return False

def find_best_model(output_name, output_dir, log_dir, save_every_n_epochs: int, max_train_epochs: int):
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={ # see below regarding this argument
         event_accumulator.COMPRESSED_HISTOGRAMS: 500,
         event_accumulator.IMAGES: 4,
         event_accumulator.AUDIO: 4,
         event_accumulator.SCALARS: 0,
         event_accumulator.HISTOGRAMS: 1,
     })
    ea.Reload()
    #get loss
    print(ea.scalars.Keys())
    print(len(ea.scalars.Items('loss/epoch')))
    items = ea.scalars.Items('loss/epoch')
    print("所有模型的损失值：", items)
    assert len(items) == max_train_epochs, "The length of items must be equal to max_train_epochs"
    best_diff = float("inf")
    best_index = None

    first_index = save_every_n_epochs - 1

    for i in range(first_index, len(items), save_every_n_epochs):
        print(i)
        diff = abs(items[i].value - 0.08)
        if diff < best_diff or (abs(diff - best_diff) < 0.005 and i > best_index):
            best_diff = diff
            best_index = i

    best_index = best_index + 1
    print("最好的模型索引是：", best_index)

    model_base_name = "{}-{{:06d}}.safetensors".format(output_name)
    model_name = model_base_name.format(best_index)
    pattern = os.path.join(output_dir, model_name)
    pattern = pattern.replace("\\", "/")
    print("最好的模型路径是：", pattern)
    # 使用 glob 模块查找匹配的文件
    files = glob.glob(pattern)

    # 如果找到一个匹配的文件，返回其路径
    if files:
        print(f"找到最好的模型：{files[0]}  {model_name}", )
        model_dir = os.path.dirname(files[0])
        return model_dir, model_name

    # 如果没有找到匹配的文件，返回 None
    print("没有找到最好的模型")
    return None


def check_training_params(data):
    potential_path = [
        "train_data_dir", "reg_data_dir", "output_dir"
    ]
    file_paths = [
        "sample_prompts"
    ]
    for p in potential_path:
        if p in data and not os.path.exists(data[p]):
            return False

    for f in file_paths:
        if f in data and not os.path.exists(data[f]):
            return False
    return True


def get_unique_folder_path(model_name: str) -> str:
    base_dir = "./train"
    count = 0
    while True:
        if count == 0:
            folder_name = model_name
        else:
            folder_name = f"{model_name}_{count}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            folder_path = folder_path.replace("\\", "/")
            return folder_name, folder_path
        count += 1


#'168529894851919052915-000002.safetensors' to '168529894851919052915.safetensors'
def replace_lora_name(filename: str):
    # 找到最后一个'.'
    last_dot_index = filename.rfind('.')
    # 如果找不到'.', 那么文件名不需要改变
    if last_dot_index == -1:
        return filename
    else:
        # 找到最后一个'-'，它在最后一个'.'前面
        last_dash_index = filename.rfind('-', 0, last_dot_index)
        # 如果找不到'-'，那么文件名不需要改变
        if last_dash_index == -1:
            return filename
        else:
            # 移除'-'和'.'之间的部分
            return filename[:last_dash_index] + filename[last_dot_index:]

def prepare_requirements():
    requirements = [
        "dadaptation", "prodigyopt"
    ]
    for r in requirements:
        if not is_installed(r):
            run_pip(f"install {r}", r)


def get_total_images(path):
    image_files = glob.glob(path + '/**/*.jpg', recursive=True)
    image_files += glob.glob(path + '/**/*.jpeg', recursive=True)
    image_files += glob.glob(path + '/**/*.png', recursive=True)
    return len(image_files)


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run(command,
        desc: Optional[str] = None,
        errdesc: Optional[str] = None,
        custom_env: Optional[list] = None,
        live: Optional[bool] = True,
        shell: Optional[bool] = False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=shell, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=shell, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(command, desc=None, live=False):
    return run(f'"{python_bin}" -m pip {command}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)

