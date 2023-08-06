import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from threading import Lock
from typing import Optional
import uuid
from typing import List
from fastapi.concurrency import run_in_threadpool
import starlette.responses as starlette_responses
from fastapi import BackgroundTasks, FastAPI, Request,Form, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import time

import mikazuki.utils as utils
import toml
from mikazuki.models import TaggerInterrogateRequest, TrainingInfo
from mikazuki.tagger.interrogator import (available_interrogators,
                                          on_interrogate)
from mikazuki.training_info_manager import train_info_manager
from mikazuki.log.logger import logger
app = FastAPI()
lock = Lock()
avaliable_scripts = [
    "networks/extract_lora_from_models.py",
    "networks/extract_lora_from_dylora.py"
]
# fix mimetype error in some fucking systems
_origin_guess_type = starlette_responses.guess_type


def _hooked_guess_type(*args, **kwargs):
    url = args[0]
    r = _origin_guess_type(*args, **kwargs)
    if url.endswith(".js"):
        r = ("application/javascript", None)
    elif url.endswith(".css"):
        r = ("text/css", None)
    return r

starlette_responses.guess_type = _hooked_guess_type


def run_train(toml_path: str,task_id:str,
              cpu_threads: Optional[int] = 2):
    print(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    print(f"开始训练：{sys.executable} os.getcwd(){os.getcwd()} os.environ{os.environ}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", str(cpu_threads),
        "./sd-scripts/train_network.py",
        "--config_file", toml_path,
    ]
    try:
        result = subprocess.run(args, env=os.environ)
        if train_info_manager.training_info_list.get(task_id) is None:
            train_info_manager.training_info_list[task_id] = TrainingInfo(error="训练完成发现该task_id不存在，奇怪的错误")
        else:
            training = train_info_manager.training_info_list[task_id]
            if result.returncode != 0:
                training.is_training=False
                training.success=False
                training.error="训练失败"
                print(f"Training failed / 训练失败")
            else:
                training.is_training=False
                training.success=True
                training.error=""
                print(f"Training finished / 训练完成")
        train_info_manager.save_training_info()

    except Exception as e:
        print(f"An error occurred when training / 创建训练进程时出现致命错误: {e}")
    finally:
        lock.release()

def run_interrogate(images_path):
    tag = TaggerInterrogateRequest(path=images_path)
    interrogator = available_interrogators.get(tag.interrogator_model, available_interrogators["wd14-convnextv2-v2"])
    on_interrogate(image=None,
                    batch_input_glob=tag.path,
                    batch_input_recursive=False,
                    batch_output_dir="",
                    batch_output_filename_format="[name].[output_extension]",
                    batch_output_action_on_conflict=tag.batch_output_action_on_conflict,
                    batch_remove_duplicated_tag=True,
                    batch_output_save_json=False,
                    interrogator=interrogator,
                    threshold=tag.threshold,
                    additional_tags=tag.additional_tags,
                    exclude_tags=tag.exclude_tags,
                    sort_by_alphabetical_order=False,
                    add_confident_as_weight=False,
                    replace_underscore=tag.replace_underscore,
                    replace_underscore_excludes=tag.replace_underscore_excludes,
                    escape_tag=tag.escape_tag,
                    unload_model_after_running=True
                    )
    return True

async def run_find_best_model(task_id:str, model_name:str, output_dir:str, logging_dir:str, save_every_n_epochs: int, max_train_epochs: int):
    try:
        if train_info_manager.training_info_list.get(task_id) is None:
            train_info_manager.training_info_list[task_id] = TrainingInfo(error="该task_id不存在，奇怪的错误")
        else:
            training = train_info_manager.training_info_list[task_id]
        tensorboard_log =  utils.find_event_file(logging_dir)
        if tensorboard_log is None:
            training.error = "虽然训练成功，但是找不到tensorboard日志文件！"
        else:
            print("tensorboard_log日志地址在",tensorboard_log)
            result = utils.find_best_model(model_name, output_dir, tensorboard_log, save_every_n_epochs, max_train_epochs)
            print("result",result)
            if result is None:
                training.error = "虽然训练成功，但是找不到最优模型！"
                print("虽然训练成功，但是找不到最优模型！")
            else:
                model_dir, new_model_name = result
                print(f"找到啦！{model_dir} {new_model_name} {model_name}")
                training.lora_path = model_dir
                #假设lora模型最好的是haha-01.safetensors, 我们先要把haha.safetensors 改成haha_old.safetensors,再将haha-01.safetensors改成haha.safetensors，前提是最好的模型不是haha.safetensors
                if new_model_name != f"{model_name}.safetensors":
                    old = f"{model_name}.safetensors"
                    new = f"{model_name}_old.safetensors"
                    res1 = utils.change_model_name(model_dir, old, new)
                    old = f"{new_model_name}"
                    new = f"{model_name}.safetensors"
                    res2 = utils.change_model_name(model_dir, old, new)
                    print("res1",res1,"res2",res2)
                    if not res1 or not res2:
                        training.error = f"最好的模型是{new_model_name}，但是改名失败"
                        print(f"最好的模型是{new_model_name}，但是改名失败")
                else:
                    print(f"最好的模型就是原模型名字，无需修改：{new_model_name}")



    except Exception as e:
        print(f"An error occurred when run_find_best_model: {e}")


async def run_process(toml_data, toml_path: str, task_id:str, images: List[UploadFile] = File(...), cpu_threads: Optional[int] = 2):
    lora_model_name = toml_data["output_name"]
    output_dir = toml_data["output_dir"]
    logging_dir = toml_data["logging_dir"]
    image_path = toml_data["train_data_dir"]
    save_every_n_epochs = toml_data["save_every_n_epochs"]
    max_train_epochs = toml_data["max_train_epochs"]
    
    image_path = os.path.join(image_path, f"50_{lora_model_name}")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for image in images:
        contents = await image.read()
        with open(os.path.join(image_path, image.filename), "wb") as f:
            f.write(contents)

    #打标签
    await run_in_threadpool(run_interrogate, image_path)

    #开始训练
    await run_in_threadpool(run_train, toml_path, task_id, cpu_threads)

    await run_find_best_model(task_id, lora_model_name, output_dir, logging_dir, save_every_n_epochs, max_train_epochs)

async def run_process_without_interrogate(toml_data, toml_path: str, task_id:str, cpu_threads: Optional[int] = 2):
    lora_model_name = toml_data["output_name"]
    output_dir = toml_data["output_dir"]
    logging_dir = toml_data["logging_dir"]
    image_path = toml_data["train_data_dir"]
    save_every_n_epochs = toml_data["save_every_n_epochs"]
    max_train_epochs = toml_data["max_train_epochs"]

    #图片已经处理过，这里将所有图片移动到子文件夹 
    image_new_path = os.path.join(image_path, f"50_{lora_model_name}")
    utils.move_images(image_path, image_new_path)

    #开始训练
    await run_in_threadpool(run_train, toml_path, task_id, cpu_threads)

    await run_find_best_model(task_id, lora_model_name, output_dir, logging_dir, save_every_n_epochs, max_train_epochs)


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response

@app.get("/training_status")
def get_training_status(task_id: str = Query(...)):
    if train_info_manager.training_info_list.get(task_id) is None:
        return {"code": -1, "msg": "请求的task_id不存在"}
    elif train_info_manager.training_info_list[task_id].is_training:
        return {"code": 1, "msg": "正在训练", "data": train_info_manager.training_info_list[task_id].dict()}
    else:
        #将训练信息转换成json格式返回
        return  {"code": 1, "msg": "训练已经完成", "data": train_info_manager.training_info_list[task_id].dict()}

@app.post("/run")
async def try_run_train_lora(background_tasks: BackgroundTasks, toml_data: str = Form(...), images: List[UploadFile] = File(None)):
    acquired = lock.acquire(blocking=False)
    
    # try:
            
    if not acquired:
        print("Training is already running / 已有正在进行的训练")
        return {"code": -1, "detail": "已有正在进行的训练"}

    #使用uuid随机生成一个taskid不能重复,字符串类型
    task_id = str(uuid.uuid1())
    while train_info_manager.training_info_list.get(task_id) is not None:
        task_id = str(uuid.uuid1())

    print(f"Training started with config file / 训练开始，使用配置文件: {toml_data}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_file = os.path.join(os.getcwd(), f"toml", "autosave", f"{timestamp}.toml")
    # 这里不需要再从请求体中读取数据，可以直接使用 `toml` 参数
    j = json.loads(toml_data)

    ok = utils.check_training_params(j)
    if not ok:
        lock.release()
        return {"code": -2, "detail": "训练目录校验失败，请确保填写的目录存在"}

    utils.prepare_requirements()
    # suggest_cpu_threads = 8 if len(images) > 100 else 2
    suggest_cpu_threads = 2

    model = j["output_name"]
    lora_model_name, image_path = utils.get_unique_folder_path(model, images is not None)
    j["train_data_dir"] = image_path
    j["output_name"] = lora_model_name

    log_prefix = "" if j.get("log_prefix") is None else j.get("log_prefix")

    logging_dir = j["logging_dir"]  + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    j["logging_dir"] = logging_dir
    old_output_dir = j["output_dir"]
    j["output_dir"] = f"{old_output_dir}/{lora_model_name}"

    train_info_manager.training_info_list[task_id] = TrainingInfo(is_training=True, success=False, error="", progress=0.0, lora_name=lora_model_name, model_dir="")

    with open(toml_file, "w") as f:
        f.write(toml.dumps(j))

    if images is None:
        background_tasks.add_task(run_process_without_interrogate, j, toml_file, task_id, suggest_cpu_threads)
    else:
        background_tasks.add_task(run_process, j, toml_file, task_id, images, suggest_cpu_threads)
    # except Exception as e:
    #     print(f"An error occurred when training / 创建训练进程时出现致命错误: {e}")
    #     lock.release()
    #     return {"code": -3, "detail": "创建训练进程时出现致命错误"}
    return {"code": 1, "task_id": task_id} #开始训练
