## 介绍

本仓库 opencompass-tpu fork 自仓库 [OpenCompass](https://github.com/open-compass/opencompass)，支持 OpenCompass 的全部功能，并支持算能 bmodel 的全流程评测。更多 OpenCompass 的更多信息请参考[官方文档](https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md)

## 🛠️ 安装指南

下面提供了快速安装的步骤。

### 💻 环境搭建

我们强烈建议使用 `conda` 来管理您的 Python 环境。

- #### 创建虚拟环境

  ```bash
  conda create --name opencompass python=3.10 -y
  conda activate opencompass
  ```

- #### 通过pip安装OpenCompass

  ```bash
  # 支持绝大多数数据集及模型
  pip install -U opencompass

  # 完整安装（支持更多数据集）
  # pip install "opencompass[full]"
  ```

- #### 准备模型
  确保本地已有 [LLM_TPU](https://github.com/sophgo/LLM-TPU) ，并完成编译所评测模型以及对话 demo。

## 🏗️ ️评测

opencompass-tpu 支持两种评测方式：配置config、使用命令行

- ### [推荐]配置 config 评测

  参考 opencompass/configs/models/qwen2_5/sg_qwen2_5_3b_instruct.py
  ```bash
    from opencompass.models import SophgoModel

    models = [
        dict(
            type=SophgoModel,
            abbr='your_model_name',
            model_path='your_bmodel_path',
            sg_tokenizer_path='your_tokenizer_path',
            max_out_len=4096,
            batch_size=1,            # only support batch_size=1
            batch_padding=False,
            device="tpu",
            devid="0",
            sample_kwargs=dict(
                temperature=1.0,
                top_p=1.0,
                repeat_penalty=1.0,
                repeat_last_n=32,
                generation_mode="greedy"
            ),
            run_cfg=dict(),
        )
    ]
  ```
  在相应位置编写完 config 后，运行：
  ```bash
  export CHAT_PATH="your_chat_demo_path"
  python run.py --datasets dataset_name --models sg_qwen2_5_3b_instruct --debug --sophgo_mode
  ```

- ### 使用命令行评测
  也可以将 config 中的配置写在命令行中，比如：
  ```bash
  python run.py \
    --datasets ceval_gen \
    --model_path your_bmodel_path \
    --sg_tokenizer_path your_tokenizer_path \
    --max-out-len 4096 \
    --device tpu \
    --devid 0 \
    --debug \
    --sophgo_mode
  ```