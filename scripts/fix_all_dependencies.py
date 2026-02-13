#!/usr/bin/env python3
"""
完整依赖修复脚本（推荐方案）
自动检测环境并修复版本不兼容问题
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并返回结果"""
    if description:
        print(f"\n[执行] {description}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=isinstance(cmd, str)
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 70)
    print("CAIL2020 法律问答 BERT - 完整依赖修复")
    print("=" * 70)

    # 第一步：检查 Python 版本
    print(f"\n[1/6] 检查 Python 版本...")
    print(f"  当前版本: {sys.version}")
    if sys.version_info < (3, 8):
        print("? Python 版本过低，需要 3.8+")
        return False
    print("? Python 版本符合要求")

    # 第二步：检查虚拟环境
    print(f"\n[2/6] 检查虚拟环境...")
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(f"? 运行在虚拟环境中")
        print(f"  环境路径: {sys.prefix}")
    else:
        print("? 不在虚拟环境中（建议使用虚拟环境）")

    # 第三步：升级 pip
    print(f"\n[3/6] 升级 pip...")
    success, stdout, stderr = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "升级 pip"
    )
    if success:
        print("? pip 升级成功")
    else:
        print("? pip 升级可能失败（继续...）")

    # 第四步：卸载旧版本
    print(f"\n[4/6] 卸载不兼容的旧版本...")
    packages_to_remove = ["peft", "accelerate"]
    for pkg in packages_to_remove:
        run_command(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            f"卸载 {pkg}"
        )
    print("? 旧版本卸载完成")

    # 第五步：安装兼容版本
    print(f"\n[5/6] 安装兼容的新版本...")
    packages = [
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "transformers>=4.35.0",
        "datasets>=2.10.0",
        "torch>=2.0.0",
    ]

    for pkg in packages:
        success, stdout, stderr = run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", pkg],
            f"安装 {pkg}"
        )
        if success:
            print(f"? {pkg} 安装成功")
        else:
            print(f"? {pkg} 安装失败")
            print(f"  错误: {stderr[:200]}")
            return False

    # 第六步：验证安装
    print(f"\n[6/6] 验证安装...")
    try:
        import transformers
        import accelerate
        import peft
        import datasets
        import torch

        print(f"? transformers: {transformers.__version__}")
        print(f"? accelerate: {accelerate.__version__}")
        print(f"? peft: {peft.__version__}")
        print(f"? datasets: {datasets.__version__}")
        print(f"? torch: {torch.__version__}")

        # 最关键的测试
        from peft import get_peft_model, LoraConfig, TaskType
        from transformers import AutoTokenizer, AutoModelForMultipleChoice
        print("\n? 关键模块导入成功")

    except ImportError as e:
        print(f"? 导入失败: {e}")
        return False

    # 完成
    print("\n" + "=" * 70)
    print("? 修复完成！所有依赖已正确安装")
    print("=" * 70)


    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
