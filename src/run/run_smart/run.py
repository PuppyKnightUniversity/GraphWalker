import subprocess
import random
import os
import sys

def _infer_nproc_from_cuda_visible_devices(cuda_visible_devices):
    """
    根据 CUDA_VISIBLE_DEVICES 计算GPU数量
    
    Args:
        cuda_visible_devices: CUDA_VISIBLE_DEVICES 环境变量的值，例如 "0,1,2,3" 或 "6,7"
    
    Returns:
        GPU数量（字符串格式）
    """
    if cuda_visible_devices:
        return str(len([d for d in cuda_visible_devices.split(',') if d.strip() != '']))
    return "1"

def run_smart_pretrain(args, logger):
    """
    运行分布式训练命令 (使用 torchrun -m)

    Args:
        args: 包含 dataset 和 seed 等参数的命名空间对象
    """
    # 生成随机端口
    logger.info("\nRunning smart pretrain...")
    random_port = random.randint(1024, 65535)

    # 设置环境变量
    env = os.environ.copy()
    # 如果环境变量中已经设置了 CUDA_VISIBLE_DEVICES，则使用环境变量的值，否则使用默认值
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    
    # 根据 CUDA_VISIBLE_DEVICES 动态计算GPU数量
    nproc_per_node = _infer_nproc_from_cuda_visible_devices(env["CUDA_VISIBLE_DEVICES"])

    # 构建命令：注意这里用了 -m
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", nproc_per_node,
        "--master_port", str(random_port),
        "-m", "run.run_smart.smart_pretrain",   # 👈 模块方式调用
        "--dataset", args.dataset,
        "--seed", str(args.seed)
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Using random port: {random_port}")
    logger.info(f"Dataset: {args.dataset}, Seed: {args.seed}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"nproc_per_node: {nproc_per_node}")
    logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")

    try:
        # 执行命令
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # 实时显示输出
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
    """
    运行分布式训练命令 (使用 torchrun -m)

    Args:
        args: 包含 dataset 和 seed 等参数的命名空间对象
    """
    # 生成随机端口
    logger.info("\nRunning smart finetune...")
    random_port = random.randint(1024, 65535)

    # 设置环境变量
    env = os.environ.copy()
    # 如果环境变量中已经设置了 CUDA_VISIBLE_DEVICES，则使用环境变量的值，否则使用默认值
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    
    # 根据 CUDA_VISIBLE_DEVICES 动态计算GPU数量
    nproc_per_node = _infer_nproc_from_cuda_visible_devices(env["CUDA_VISIBLE_DEVICES"])

    # 构建命令：注意这里用了 -m
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", nproc_per_node,
        "--master_port", str(random_port),
        "-m", "run.run_smart.smart_finetune",   # 👈 模块方式调用
        "--dataset", args.dataset,
        "--seed", str(args.seed)
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Using random port: {random_port}")
    logger.info(f"Dataset: {args.dataset}, Seed: {args.seed}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"nproc_per_node: {nproc_per_node}")
    logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")

    try:
        # 执行命令
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # 实时显示输出
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
