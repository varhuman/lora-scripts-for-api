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

import starlette.responses as starlette_responses
from fastapi import BackgroundTasks, FastAPI, Request,Form, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import mikazuki.utils as utils
import toml
from mikazuki.models import TaggerInterrogateRequest, TrainingInfo
from mikazuki.tagger.interrogator import (available_interrogators,
                                          on_interrogate)

training_info_list: dict[str, TrainingInfo] = {} # {task_id, TrainingInfo}
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
    args = [
        sys.executable, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", str(cpu_threads),
        "./sd-scripts/train_network.py",
        "--config_file", toml_path,
    ]
    try:
        result = subprocess.run(args, env=os.environ)
        if training_info_list.get(task_id) is None:
            training_info_list[task_id] = TrainingInfo(error="该task_id不存在，奇怪的错误")
        else:
            training = training_info_list[task_id]
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
    except Exception as e:
        print(f"An error occurred when training / 创建训练进程时出现致命错误: {e}")
    finally:
        lock.release()

async def run_interrogate(images_path):
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



async def pre_train(lora_model_name: str,folder_path:str, toml_path: str, task_id:str, images: List[UploadFile] = File(...), cpu_threads: Optional[int] = 2):
    folder_path = os.path.join(folder_path, f"6_{lora_model_name}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for image in images:
        contents = await image.read()
        with open(os.path.join(folder_path, image.filename), "wb") as f:
            f.write(contents)
    await run_interrogate(folder_path)
    # Call run_train method
    run_train(toml_path,task_id, cpu_threads)


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response

@app.get("/api/training_status")
async def get_training_status(task_id: str = Query(...)):
    if training_info_list.get(task_id) is None:
        return {"code": -1, "msg": "请求的task_id不存在"}
    elif training_info_list[task_id].is_training:
        return {"code": 0, "msg": "正在训练"}
    else:
        #将训练信息转换成json格式返回
        return  {"code": 1, "msg": "训练已经完成", "data": training_info_list[task_id].dict()}

@app.post("/api/run")
async def try_run_train_lora(background_tasks: BackgroundTasks, toml_data: str = Form(...), images: List[UploadFile] = File(...)):
    acquired = lock.acquire(blocking=False)

    if not acquired:
        print("Training is already running / 已有正在进行的训练")
        return {"code": -1, "detail": "已有正在进行的训练"}

    #使用uuid随机生成一个taskid不能重复,字符串类型
    task_id = str(uuid.uuid1())
    while training_info_list.get(task_id) is not None:
        task_id = str(uuid.uuid1())

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_file = os.path.join(os.getcwd(), f"toml", "autosave", f"{timestamp}.toml")

    # 这里不需要再从请求体中读取数据，可以直接使用 `toml` 参数
    j = json.loads(toml_data)

    ok = utils.check_training_params(j)
    if not ok:
        lock.release()
        return {"code": -2, "detail": "训练目录校验失败，请确保填写的目录存在"}

    utils.prepare_requirements()
    suggest_cpu_threads = 8 if len(images) > 100 else 2

    model = j["output_name"]
    lora_model_name, folder_path = utils.get_unique_folder_path(model)
    j["train_data_dir"] = folder_path
    j["output_name"] = lora_model_name
    with open(toml_file, "w") as f:
        f.write(toml.dumps(j))
    background_tasks.add_task(pre_train,lora_model_name, folder_path, toml_file,task_id, images, suggest_cpu_threads)

    training_info_list[task_id] = True

    return {"code": 1, "task_id": task_id} #开始训练
