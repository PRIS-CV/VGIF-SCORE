import os
import sys
import json
import argparse
import datetime
import traceback
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.environ.get("LONGCAT_PROJECT_ROOT")
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_jobs(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pipeline(checkpoint_dir: str, cp_split_hw, enable_compile: bool, local_rank: int):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir,
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        torch_dtype=torch.bfloat16,
    )
    if enable_compile:
        dit = torch.compile(dit)
    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)
    return pipe


def run_jobs(args):
    jobs = load_jobs(args.jobs_json)
    if not jobs:
        print("[INFO] No pending prompts to run.", flush=True)
        return

    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA device visible to torchrun process")
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[RANK {global_rank}] local_rank={local_rank} world_size={world_size} project_root={PROJECT_ROOT}", flush=True)
    print(f"[RANK {global_rank}] checkpoint_dir={args.checkpoint_dir}", flush=True)
    print(f"[RANK {global_rank}] loading models...", flush=True)

    init_context_parallel(
        context_parallel_size=args.context_parallel_size,
        global_rank=global_rank,
        world_size=world_size,
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    pipe = build_pipeline(args.checkpoint_dir, cp_split_hw, args.enable_compile, local_rank)

    seed_base = int(os.environ.get("LONGCAT_GLOBAL_SEED", "42"))
    height = int(os.environ.get("LONGCAT_HEIGHT", "512"))
    width = int(os.environ.get("LONGCAT_WIDTH", "896"))
    num_frames = int(os.environ.get("LONGCAT_NUM_FRAMES", "93"))
    num_steps = int(os.environ.get("LONGCAT_NUM_STEPS", "50"))
    guidance_scale = float(os.environ.get("LONGCAT_GUIDANCE_SCALE", "4.0"))
    negative_prompt = os.environ.get("LONGCAT_NEGATIVE_PROMPT", "")
    fps = int(os.environ.get("LONGCAT_FPS", "15"))
    crf = str(os.environ.get("LONGCAT_CRF", "18"))

    for job in jobs:
        index = int(job["index"])
        prompt = job["prompt"]
        output_path = job["output_path"]
        prompt_txt = job["prompt_txt"]
        sample_log = job["sample_log"]

        if global_rank == 0:
            Path(prompt_txt).write_text(prompt + "\n", encoding="utf-8")
            with open(sample_log, "a", encoding="utf-8") as f:
                f.write(f"[RUN ] {index} -> {os.path.basename(output_path)}\n")
            print(f"[RUN ] {index} -> {os.path.basename(output_path)}", flush=True)

        dist.barrier()

        generator = torch.Generator(device=local_rank)
        generator.manual_seed(seed_base + index * 1000 + global_rank)

        output = pipe.generate_t2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )[0]

        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(output))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video(output_path, output_tensor, fps=fps, video_codec="libx264", options={"crf": crf})
            with open(sample_log, "a", encoding="utf-8") as f:
                f.write(f"[DONE] {index} -> {os.path.basename(output_path)}\n")

        del output
        torch_gc()
        dist.barrier()

    dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs_json", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--enable_compile", action="store_true")
    return parser.parse_args()


@record
def main():
    try:
        args = _parse_args()
        run_jobs(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
