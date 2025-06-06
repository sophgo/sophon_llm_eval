from opencompass.models import SophgoModel

models = [
    dict(
        type=SophgoModel,
        abbr='qwen2.5-3b-instruct-sophgo-awq',
        model_path='/data/romainzhou/LLM-TPU/models/Qwen2_5/python_demo/qwen2.5-3b-instruct-awq_w4bf16_seq4096_bm1684x_1dev_20250604_220747.bmodel',
        sg_tokenizer_path='/data/romainzhou/LLM-TPU/models/Qwen2_5/support/token_config/',
        max_out_len=4096,
        batch_size=1,
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
