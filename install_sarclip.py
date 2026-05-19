"""
SARCLIP 安装检查和设置脚本
"""

import sys
import subprocess
import importlib.util

def check_and_install_sarclip():
    """检查并安装SARCLIP"""
    print("检查SARCLIP是否已安装...")
    
    # 尝试导入sar_clip
    try:
        import sar_clip
        print("✓ SARCLIP 已安装")
        return True
    except ImportError:
        print("× SARCLIP 未安装，开始安装...")
        
        # 尝试通过pip安装
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sarclip"])
            print("✓ SARCLIP 安装成功")
            return True
        except subprocess.CalledProcessError:
            print("× SARCLIP pip安装失败")
            
            # 如果pip安装失败，提示用户手动安装
            print("\n请按照以下步骤安装SARCLIP:")
            print("1. 从官方仓库下载SARCLIP代码:")
            print("   git clone https://github.com/username/sarclip.git  # 替换为实际仓库地址")
            print("2. 进入目录并安装:")
            print("   cd sarclip")
            print("   pip install -e .")
            print("\n或者，如果您有本地的SARCLIP包，请确保其路径已添加到PYTHONPATH中。")
            return False

def check_sarclip_components():
    """检查SARCLIP的关键组件"""
    print("\n检查SARCLIP关键组件...")
    
    try:
        import sar_clip
        
        # 检查关键函数是否存在
        required_functions = [
            'create_model_with_args',
            'get_tokenizer', 
            'build_zero_shot_classifier'
        ]
        
        for func_name in required_functions:
            if hasattr(sar_clip, func_name):
                print(f"✓ {func_name} 可用")
            else:
                print(f"× {func_name} 不可用")
                
        return True
    except ImportError:
        print("× 无法导入sar_clip")
        return False

def main():
    print("SARCLIP 安装检查工具")
    print("="*40)
    
    # 检查并安装SARCLIP
    if check_and_install_sarclip():
        # 检查组件
        check_sarclip_components()
        
        print("\n✓ SARCLIP 环境检查完成")
        print("现在您可以运行训练脚本了。")
        return True
    else:
        print("\n× SARCLIP 环境配置失败")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)