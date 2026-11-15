import subprocess
import random
import os
import sys

def _dimension_args(args):
    result = []
    for name in ('smart_input_dim', 'smart_demo_dim', 'smart_num_class'):
        value = getattr(args, name, None)
        if value is not None:
            result.extend(['--' + name, str(value)])
    return result

def _infer_nproc_from_cuda_visible_devices(cuda_visible_devices):
    if cuda_visible_devices:
        return str(len([d for d in cuda_visible_devices.split(',') if d.strip() != '']))
    return "1"

def run_smart_pretrain(args, logger):
    logger.info("\nRunning smart pretrain...")
    random_port = random.randint(1024, 65535)

    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    
    nproc_per_node = _infer_nproc_from_cuda_visible_devices(env["CUDA_VISIBLE_DEVICES"])

    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", nproc_per_node,
        "--master_port", str(random_port),
        "-m", "run.run_smart.smart_pretrain",
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        *_dimension_args(args),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Using random port: {random_port}")
    logger.info(f"Dataset: {args.dataset}, Seed: {args.seed}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"nproc_per_node: {nproc_per_node}")
    logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info("Smart pretrain completed successfully!\n")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred during smart pretrain: {e}")
        logger.error(f"Return code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during smart pretrain: {e}")
        sys.exit(1)

def run_smart_finetune(args, logger):
    logger.info("\nRunning smart finetune...")
    random_port = random.randint(1024, 65535)

    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    
    nproc_per_node = _infer_nproc_from_cuda_visible_devices(env["CUDA_VISIBLE_DEVICES"])

    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", nproc_per_node,
        "--master_port", str(random_port),
        "-m", "run.run_smart.smart_finetune",
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        *_dimension_args(args),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Using random port: {random_port}")
    logger.info(f"Dataset: {args.dataset}, Seed: {args.seed}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"nproc_per_node: {nproc_per_node}")
    logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info("Smart finetune completed successfully!\n")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred during smart finetune: {e}")
        logger.error(f"Return code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during smart finetune: {e}")
        sys.exit(1)

def run(args, train_dataset, val_dataset, test_dataset, logger):
    run_smart_pretrain(args, logger)
    run_smart_finetune(args, logger)
