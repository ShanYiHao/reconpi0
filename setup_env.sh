#!/bin/bash

# 设置错误时立即退出
set -e

# 定义颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}>>> 开始配置 Pi0-Recon-VLA 运行环境...${NC}"

# 1. 检查是否在项目根目录
if [ ! -d "src/pi0_core/transformers_replace" ]; then
    echo -e "${RED}错误: 请在项目根目录下运行此脚本！${NC}"
    echo "当前路径缺少 src/pi0_core/transformers_replace 文件夹。"
    exit 1
fi

# 2. 安装 Python 依赖
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}>>> 正在安装/更新 Python 依赖 (pip)...${NC}"
    pip install -r requirements.txt
else
    echo -e "${RED}错误: 未找到 requirements.txt 文件。${NC}"
    exit 1
fi

# 3. 执行 Transformers 库补丁 (最关键步骤)
echo -e "${YELLOW}>>> 正在应用 OpenPI 的 Transformers 底层补丁...${NC}"

# 使用 Python 动态获取当前环境 transformers 的安装路径
TRANSFORMERS_DIR=$(python -c "import transformers; import os; print(transformers.__path__[0])")
MODELS_DIR="$TRANSFORMERS_DIR/models"

if [ -d "$MODELS_DIR" ]; then
    echo "定位到 Transformers 安装路径: $TRANSFORMERS_DIR"
    
    # 执行覆盖操作
    # cp -r src/pi0_core/transformers_replace/models/* target_dir/
    cp -r src/pi0_core/transformers_replace/models/* "$MODELS_DIR/"
    
    echo -e "${GREEN}✅ 补丁应用成功！已覆盖 OpenPI 修改版的 SigLIP/Gemma 定义。${NC}"
else
    echo -e "${RED}❌ 严重错误: 无法找到 transformers 库。${NC}"
    echo "请检查 'pip install transformers==4.53.2' 是否执行成功。"
    exit 1
fi

# 4. 验证环境
echo -e "${YELLOW}>>> 正在进行最终环境验证...${NC}"
python -c "import torch; import transformers; import peft; print(f'环境验证通过:\nTorch: {torch.__version__}\nTransformers: {transformers.__version__} (Patch Applied)')"

echo -e "${GREEN}🎉 环境配置全部完成！${NC}"
echo -e "请运行以下命令进行冒烟测试："
echo -e "${YELLOW}python debug_model.py${NC}"