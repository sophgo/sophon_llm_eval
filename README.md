## Introduction
The sophon_llm_eval is a fork of [OpenCompass](https://github.com/open-compass/opencompass). It supports all OpenCompass features and provides end-to-end evaluation for Sophgo bmodel. For more details on OpenCompass, please refer to the [official documentation](https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md).


## 🛠️ Installation Guide
Below is a quick install walkthrough.

### 💻 Environment Setup
We strongly recommend using conda to manage your Python environment.

- #### Create a virtual environment
  ```bash
  conda create --name opencompass python=3.10 -y
  conda activate opencompass
  ```
- #### Install OpenCompass via pip
  ```bash
  # Supports most datasets and models
  pip install -U opencompass

  # Full install (supports additional datasets)
  # pip install "opencompass[full]"
  ```

- #### Prepare the model
  Make sure you have the [LLM_TPU](https://github.com/sophgo/LLM-TPU) repository locally, and that your target bmodel and chat demo are compiled.


## 🏗️ Evaluation
sophon_llm_eval supports two evaluation modes: config-based and CLI-based.

- ### [Recommended] Config-based Evaluation
  Refer to opencompass/configs/models/qwen2_5/sg_qwen2_5_3b_instruct.py:
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
  After writing your config, run:
  ```bash
  export CHAT_PATH="your_chat_demo_path"
  python run.py --datasets dataset_name --models sg_qwen2_5_3b_instruct --debug --sophgo_mode
  ```
- ### CLI-based Evaluation
  You can also pass the same settings directly on the command line, for example:

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