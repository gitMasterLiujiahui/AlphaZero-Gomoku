"""
Installation Script for Gomoku AI
五子棋AI安装脚本
"""

import subprocess
import sys
import os


def install_requirements():
    """安装依赖包"""
    print("Installing required packages...")
    
    try:
        # 升级pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 安装依赖
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✓ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install packages: {e}")
        return False


def test_installation():
    """测试安装"""
    print("Testing installation...")
    
    try:
        import torch
        import pygame
        import numpy
        print("✓ All required packages are available!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("Gomoku AI Installation Script")
    print("=" * 50)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required!")
        return 1
    
    print(f"✓ Python version: {sys.version}")
    
    # 安装依赖
    if not install_requirements():
        return 1
    
    # 测试安装
    if not test_installation():
        return 1
    
    print("=" * 50)
    print("Installation completed successfully!")
    print("You can now run: python main.py")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
