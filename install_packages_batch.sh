#!/bin/bash
# 分批安装 Python 包以避免内存不足问题
# 使用方法: bash install_packages_batch.sh

set -e  # 遇到错误立即退出

ENV_NAME="ehrbase"
echo "开始分批安装包到环境: $ENV_NAME"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 检查环境是否激活
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "错误: 无法激活 conda 环境 $ENV_NAME"
    exit 1
fi

# 函数：安装一批包
install_batch() {
    local batch_name=$1
    shift
    local packages=("$@")
    
    echo ""
    echo "=========================================="
    echo "安装批次: $batch_name"
    echo "包数量: ${#packages[@]}"
    echo "=========================================="
    
    # 构建 pip install 命令
    local install_cmd="pip install --no-cache-dir"
    for pkg in "${packages[@]}"; do
        install_cmd="$install_cmd $pkg"
    done
    
    echo "执行: $install_cmd"
    if $install_cmd; then
        echo "✓ $batch_name 安装成功"
    else
        echo "✗ $batch_name 安装失败"
        exit 1
    fi
    
    # 清理 pip 缓存以释放内存
    pip cache purge
    echo "已清理 pip 缓存"
}

# 第二批：PyTorch 核心包（需要先安装，其他包依赖它）
echo ""
echo "=========================================="
echo "第二批：PyTorch 核心包"
echo "=========================================="
install_batch "PyTorch核心包" \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0"

# 第三批：CUDA 相关包（大型二进制包）
echo ""
echo "=========================================="
echo "第三批：CUDA 相关包"
echo "=========================================="
install_batch "CUDA相关包" \
    "nvidia-cublas-cu12==12.4.5.8" \
    "nvidia-cuda-cupti-cu12==12.4.127" \
    "nvidia-cuda-nvrtc-cu12==12.4.127" \
    "nvidia-cuda-runtime-cu12==12.4.127" \
    "nvidia-cudnn-cu12==9.1.0.70" \
    "nvidia-cufft-cu12==11.2.1.3" \
    "nvidia-cufile-cu12==1.11.1.6" \
    "nvidia-curand-cu12==10.3.5.147" \
    "nvidia-cusolver-cu12==11.6.1.9" \
    "nvidia-cusparse-cu12==12.3.1.170" \
    "nvidia-cusparselt-cu12==0.6.2" \
    "nvidia-ml-py==12.575.51" \
    "nvidia-nccl-cu12==2.21.5" \
    "nvidia-nvjitlink-cu12==12.4.127" \
    "nvidia-nvtx-cu12==12.4.127"

# 第四批：科学计算基础包
echo ""
echo "=========================================="
echo "第四批：科学计算基础包"
echo "=========================================="
install_batch "科学计算基础包" \
    "numpy==1.26.4" \
    "scipy==1.15.2" \
    "pandas==1.5.3" \
    "scikit-learn==1.6.1"

# 第五批：编译密集型包（需要编译，占用内存）
echo ""
echo "=========================================="
echo "第五批：编译密集型包（分批安装）"
echo "=========================================="

# 5.1: 先安装编译工具
install_batch "编译工具" \
    "ninja==1.11.1.4"

# 5.2: numba 和 llvmlite（需要编译）
install_batch "Numba相关" \
    "llvmlite==0.44.0" \
    "numba==0.61.2"

# 5.3: triton（需要编译，大型包）
install_batch "Triton" \
    "triton==3.2.0"

# 5.4: cupy（需要编译，非常占用内存）
echo ""
echo "警告: cupy 编译会占用大量内存，请确保有足够内存"
install_batch "CuPy" \
    "cupy-cuda12x==13.4.1"

# 第六批：深度学习框架核心包
echo ""
echo "=========================================="
echo "第六批：深度学习框架核心包"
echo "=========================================="
install_batch "深度学习框架核心" \
    "transformers==4.51.3" \
    "accelerate==1.6.0" \
    "tokenizers==0.21.1" \
    "safetensors==0.5.3" \
    "huggingface-hub==0.30.2" \
    "tiktoken==0.9.0"

# 第七批：大型深度学习包（分批安装）
echo ""
echo "=========================================="
echo "第七批：大型深度学习包"
echo "=========================================="

# 7.1: deepspeed（需要编译）
echo ""
echo "警告: deepspeed 编译会占用大量内存"
install_batch "DeepSpeed" \
    "deepspeed==0.16.7"

# 7.2: xformers（需要编译）
install_batch "xformers" \
    "xformers==0.0.29.post2"

# 7.3: vllm（大型包）
echo ""
echo "警告: vllm 安装会占用大量内存"
install_batch "vLLM" \
    "vllm==0.8.5.post1"

# 第八批：其他大型包
echo ""
echo "=========================================="
echo "第八批：其他大型包"
echo "=========================================="
install_batch "其他大型包" \
    "ray==2.46.0" \
    "datasets==3.5.0" \
    "gradio==5.25.0" \
    "gradio-client==1.8.0" \
    "peft==0.15.1" \
    "trl==0.9.6" \
    "compressed-tensors==0.9.3" \
    "tensordict==0.10.0"

# 第九批：音频和图像处理包
echo ""
echo "=========================================="
echo "第九批：音频和图像处理包"
echo "=========================================="
install_batch "音频图像处理" \
    "librosa==0.11.0" \
    "soundfile==0.13.1" \
    "soxr==0.5.0.post1" \
    "audioread==3.0.1" \
    "av==14.3.0" \
    "pydub==0.25.1" \
    "ffmpy==0.5.0" \
    "opencv-python-headless==4.11.0.86" \
    "pillow==11.2.1" \
    "matplotlib==3.10.1" \
    "contourpy==1.3.2"

# 第十批：Web 框架和 API
echo ""
echo "=========================================="
echo "第十批：Web 框架和 API"
echo "=========================================="
install_batch "Web框架" \
    "fastapi==0.115.12" \
    "fastapi-cli==0.0.7" \
    "uvicorn==0.34.2" \
    "prometheus-client==0.22.0" \
    "prometheus-fastapi-instrumentator==7.1.0"

# 第十一批：OpenTelemetry 相关
echo ""
echo "=========================================="
echo "第十一批：OpenTelemetry 相关"
echo "=========================================="
install_batch "OpenTelemetry" \
    "opentelemetry-api==1.26.0" \
    "opentelemetry-proto==1.26.0" \
    "opentelemetry-sdk==1.26.0" \
    "opentelemetry-exporter-otlp==1.26.0" \
    "opentelemetry-exporter-otlp-proto-common==1.26.0" \
    "opentelemetry-exporter-otlp-proto-grpc==1.26.0" \
    "opentelemetry-exporter-otlp-proto-http==1.26.0" \
    "opentelemetry-semantic-conventions==0.47b0" \
    "opentelemetry-semantic-conventions-ai==0.4.9"

# 第十二批：其他中型包
echo ""
echo "=========================================="
echo "第十二批：其他中型包"
echo "=========================================="
install_batch "其他中型包" \
    "antlr4-python3-runtime==4.9.3" \
    "einops==0.8.1" \
    "depyf==0.18.0" \
    "fastrlock==0.8.3" \
    "fire==0.7.0" \
    "pandarallel==1.6.5" \
    "multiprocess==0.70.16" \
    "pyarrow==20.0.0" \
    "pydantic==2.10.6" \
    "pydantic-core==2.27.2" \
    "protobuf==4.25.7" \
    "grpcio==1.71.0" \
    "psutil==7.0.0" \
    "pyzmq==26.4.0"

# 第十三批：专业库
echo ""
echo "=========================================="
echo "第十三批：专业库"
echo "=========================================="
install_batch "专业库" \
    "mne==1.9.0" \
    "rdkit==2025.3.2" \
    "igraph==1.0.0" \
    "leidenalg==0.11.0" \
    "pyhealth==1.1.6"

# 第十四批：AI 和 LLM 相关
echo ""
echo "=========================================="
echo "第十四批：AI 和 LLM 相关"
echo "=========================================="
install_batch "AI和LLM相关" \
    "openai==1.79.0" \
    "modelscope==1.25.0" \
    "mistral-common==1.5.6" \
    "outlines==0.1.11" \
    "outlines-core==0.1.26" \
    "lm-format-enforcer==0.10.11" \
    "llguidance==0.7.24" \
    "xgrammar==0.1.18" \
    "gguf==0.16.3" \
    "hf-xet==1.1.2"

# 第十五批：工具和开发包
echo ""
echo "=========================================="
echo "第十五批：工具和开发包"
echo "=========================================="
install_batch "工具和开发包" \
    "ruff==0.11.8" \
    "pdbpp==0.11.7" \
    "rich==14.0.0" \
    "rich-toolkit==0.14.6" \
    "typer==0.15.3" \
    "tyro==0.8.14" \
    "jsonschema==4.23.0" \
    "safehttpx==0.1.6"

# 完成
echo ""
echo "=========================================="
echo "✓ 所有包安装完成！"
echo "=========================================="
echo ""
echo "验证安装:"
pip list | grep -E "(torch|transformers|vllm|deepspeed)" || echo "部分包可能未安装，请检查"

