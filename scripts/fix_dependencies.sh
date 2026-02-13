#!/bin/bash
# 一键修复依赖问题

echo "============================================================"
echo "CAIL2020 法律问答 - 依赖修复脚本"
echo "============================================================"
echo ""

# 获取当前环境信息
echo "[信息] 检查当前环境..."
python --version
echo ""

# 升级 pip
echo "[步骤 1/4] 升级 pip..."
pip install --upgrade pip > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "? pip 升级成功"
else
    echo "? pip 升级失败（可能无法升级，继续...）"
fi
echo ""

# 卸载旧版本的不兼容包
echo "[步骤 2/4] 清理不兼容的旧版本包..."
pip uninstall -y peft accelerate > /dev/null 2>&1
echo "? 旧包清理完成"
echo ""

# 安装兼容的新版本
echo "[步骤 3/4] 安装兼容的新版本..."
pip install --upgrade \
    "peft>=0.7.0" \
    "accelerate>=0.25.0" \
    "transformers>=4.35.0"

if [ $? -eq 0 ]; then
    echo "? 新版本安装成功"
else
    echo "? 安装失败，请检查网络连接"
    exit 1
fi
echo ""

# 验证安装
echo "[步骤 4/4] 验证安装..."
python -c "
import transformers
import accelerate
import peft
print(f'  transformers: {transformers.__version__}')
print(f'  accelerate: {accelerate.__version__}')
print(f'  peft: {peft.__version__}')
print('? 所有包安装成功')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "? 修复完成！现在可以运行训练："
    echo "============================================================"
else
    echo ""
    echo "? 验证失败，可能存在其他问题"
    echo "请尝试运行: python fix_dependencies.py"
fi
