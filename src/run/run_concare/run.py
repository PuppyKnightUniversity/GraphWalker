import os
import sys
import subprocess
import random
import logging

def _infer_nproc_from_cuda_visible_devices(cuda_visible_devices):
    if cuda_visible_devices:
        return str(len([d for d in cuda_visible_devices.split(',') if d.strip() != '']))
    return "1"

def run_concare_train(args, logger):
    logger.info("\nRunning ConCare training...")
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
        "-m", "run.run_concare.concare_train",
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        # Forward ConCare training options to the subprocess.
        "--concare_save_dir", args.concare_save_dir,
        "--concare_batch_size", str(args.concare_batch_size),
        "--concare_epochs", str(args.concare_epochs),
        "--concare_lr", str(args.concare_lr),
        "--concare_dropout", str(args.concare_dropout),
        "--concare_hidden_dim", str(args.concare_hidden_dim),
        "--concare_num_head", str(args.concare_num_head),
    ]

    if args.toy_dataset:
        cmd.append("--toy_dataset")

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info("ConCare training completed successfully!\n")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred during ConCare training: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during ConCare training: {e}")
        sys.exit(1)

def run(args, train_dataset, val_dataset, test_dataset, logger):
    # The training subprocess loads its own datasets.
    run_concare_train(args, logger)

if __name__ == "__main__":
    from args.ehrbase_args import parse_args
    import pickle

    args = parse_args()

    # Load cached datasets for direct module execution.
    train_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_train.pkl', 'rb'))
    val_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_val.pkl', 'rb'))
    test_dataset = pickle.load(open(args.mid_data_dump_path + f'/{args.dataset}/seed' + str(args.seed) + f'/{args.dataset}_test.pkl', 'rb'))

    if getattr(args, 'toy_dataset', False):
        print("Using toy dataset...")
        train_dataset = {key: value[:1600] for key, value in train_dataset.items()}
        val_dataset = {key: value[:200] for key, value in val_dataset.items()}
        test_dataset = {key: value[:200] for key, value in test_dataset.items()}

    from run.run_concare.concare_train import concare_train
    concare_train(args, train_dataset, val_dataset, test_dataset)
