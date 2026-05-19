"""
SARCLIP 环境检查脚本
"""

import sys

def check_sarclip():
    """检查仓库内置 SARCLIP 是否可导入"""
    print("检查SARCLIP是否可导入...")
    
    try:
        import sar_clip
        print(f"✓ SARCLIP 可用: {sar_clip.__file__}")
        return True
    except ImportError as exc:
        print(f"× SARCLIP 导入失败: {exc}")
        print("请确认当前工作目录是 IRAOD-New，或已将仓库根目录加入 PYTHONPATH。")
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
    print("SARCLIP 环境检查工具")
    print("="*40)
    
    if check_sarclip():
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
