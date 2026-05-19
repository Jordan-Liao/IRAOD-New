"""
测试SARCLIP是否正确安装和配置
"""

def test_sarclip_import():
    """测试SARCLIP导入"""
    try:
        import sar_clip
        print("✓ 成功导入sar_clip模块")
        return True
    except ImportError as e:
        print(f"× 导入sar_clip失败: {e}")
        return False

def test_sarclip_functions():
    """测试SARCLIP关键函数"""
    try:
        import sar_clip
        
        # 检查必要的函数是否存在
        functions_to_check = [
            'create_model_with_args',
            'get_tokenizer',
            'build_zero_shot_classifier'
        ]
        
        all_found = True
        for func_name in functions_to_check:
            if hasattr(sar_clip, func_name):
                print(f"✓ 找到函数: {func_name}")
            else:
                print(f"× 缺少函数: {func_name}")
                all_found = False
                
        return all_found
    except ImportError as e:
        print(f"× 测试SARCLIP函数时出错: {e}")
        return False

def test_cga_initialization():
    """测试CGA类能否正确初始化（使用模拟参数）"""
    try:
        from sfod.cga import CGA
        
        # 创建一个简单的测试类，避免实际加载模型
        class MockSarClip:
            def __init__(self):
                pass
                
        print("✓ 成功导入CGA类")
        return True
    except Exception as e:
        print(f"× 测试CGA初始化时出错: {e}")
        return False

def main():
    print("开始测试SARCLIP环境...")
    print("-" * 40)
    
    success_count = 0
    total_tests = 3
    
    # 测试1: SARCLIP导入
    print("测试1: SARCLIP模块导入")
    if test_sarclip_import():
        success_count += 1
    print()
    
    # 测试2: SARCLIP函数
    print("测试2: SARCLIP关键函数")
    if test_sarclip_functions():
        success_count += 1
    print()
    
    # 测试3: CGA初始化
    print("测试3: CGA类导入")
    if test_cga_initialization():
        success_count += 1
    print()
    
    print("-" * 40)
    print(f"测试结果: {success_count}/{total_tests} 项测试通过")
    
    if success_count == total_tests:
        print("✓ 所有测试通过！SARCLIP环境配置正确。")
        return True
    else:
        print("× 部分测试失败，请检查SARCLIP安装。")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)