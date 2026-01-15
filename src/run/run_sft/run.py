import subprocess
import random
import os
import sys

def _infer_nproc_from_env():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        return str(len([d for d in devices.split(',') if d.strip() != '']))
    return "1"

def run(args, train_dataset, val_dataset, test_dataset, logger):
    logger.info("\nRunning SFT training...")

    nproc = _infer_nproc_from_env()

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", nproc,
        "--mixed_precision", "fp16" if getattr(args, "sft_fp16", False) else "no",
    ]
    # optional main process port to avoid conflicts when launching multiple trainings
    main_port = getattr(args, 'dist_main_port', None)
    if main_port is not None:
        cmd.extend(["--main_process_port", str(main_port)])
    cmd.extend([
        "-m", "run.run_sft.sft_train",
        "--dataset", args.dataset,
        "--seed", str(args.seed),
    ])

    # Forward essential LLM/SFT arguments to the training module
    def add_flag(name, value, is_bool=False):
        if is_bool:
            if bool(value):
                cmd.extend([f"--{name}"])
        else:
            if value is not None:
                cmd.extend([f"--{name}", str(value)])

    add_flag("llm_name", getattr(args, "llm_name", None))
    add_flag("llm_local_path", getattr(args, "llm_local_path", None))
    add_flag("sft_epochs", getattr(args, "sft_epochs", None))
    add_flag("sft_lr", getattr(args, "sft_lr", None))
    add_flag("sft_batch_size", getattr(args, "sft_batch_size", None))
    add_flag("sft_gradient_accumulation", getattr(args, "sft_gradient_accumulation", None))
    add_flag("sft_warmup_ratio", getattr(args, "sft_warmup_ratio", None))
    add_flag("sft_fp16", getattr(args, "sft_fp16", False), is_bool=True)
    add_flag("sft_max_seq_length", getattr(args, "sft_max_seq_length", None))
    add_flag("sft_output_dir", getattr(args, "sft_output_dir", None))
    add_flag("sft_use_lora", getattr(args, "sft_use_lora", False), is_bool=True)
    add_flag("sft_lora_r", getattr(args, "sft_lora_r", None))
    add_flag("sft_lora_alpha", getattr(args, "sft_lora_alpha", None))
    add_flag("sft_lora_dropout", getattr(args, "sft_lora_dropout", None))
    add_flag("sft_lora_target_modules", getattr(args, "sft_lora_target_modules", None))
    add_flag("toy_dataset", getattr(args, "toy_dataset", False), is_bool=True)
    add_flag("dist_main_port", getattr(args, "dist_main_port", None))
    add_flag("sft_gradient_checkpointing", getattr(args, "sft_gradient_checkpointing", True), is_bool=True)
    add_flag("sft_deepspeed_zero3", getattr(args, "sft_deepspeed_zero3", True), is_bool=True)
    add_flag("sft_zero3_cpu_offload", getattr(args, "sft_zero3_cpu_offload", False), is_bool=True)

    env = os.environ.copy()
    src_path = os.path.join(os.getcwd(), "src")
    env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TF_USE_LEGACY_KERAS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Dataset: {args.dataset}, Seed: {args.seed}")
    logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True,
        )
        logger.info("SFT training completed successfully!\n")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred during SFT training: {e}")
        logger.error(f"Return code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during SFT training: {e}")
        sys.exit(1)
