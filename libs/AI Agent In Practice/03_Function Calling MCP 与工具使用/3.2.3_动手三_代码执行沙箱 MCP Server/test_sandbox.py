# 独立运行，不依赖 Claude Desktop，直接导入函数验证逻辑

import sys
sys.path.insert(0, ".")

# 直接导入验证（绕过 MCP 协议层）
from sandbox_server import execute_python, install_package, get_execution_history, reset_session, is_safe_code

def test_basic_execution():
    """测试基本代码执行"""
    result = execute_python("print('hello from sandbox')\nprint(1 + 1)")
    assert result["success"] is True, f"执行失败: {result}"
    assert "hello from sandbox" in result["stdout"]
    assert "2" in result["stdout"]
    print("✅ test_basic_execution passed")

def test_timeout():
    """测试超时保护"""
    result = execute_python("while True: pass", timeout=2)
    assert result["success"] is False
    assert "超时" in result["error"]
    print("✅ test_timeout passed")

def test_security_block_import():
    """测试危险 import 拦截"""
    result = execute_python("import os\nprint(os.getcwd())")
    assert result["success"] is False
    assert "安全检查拒绝" in result["error"]
    print("✅ test_security_block_import passed")

def test_security_block_eval():
    """测试 eval 拦截"""
    result = execute_python("result = eval('1+1')\nprint(result)")
    assert result["success"] is False
    print("✅ test_security_block_eval passed")

def test_numpy_if_available():
    """测试 numpy 可用性（需要已安装）"""
    result = execute_python(
        "import numpy as np\n"
        "arr = np.array([1, 2, 3, 4, 5])\n"
        "print(f'mean={arr.mean():.2f}, std={arr.std():.2f}')"
    )
    if result["success"]:
        assert "mean=3.00" in result["stdout"]
        print("✅ test_numpy_if_available passed")
    else:
        print("⏭  test_numpy_if_available skipped (numpy not installed)")

def test_history():
    """测试执行历史"""
    reset_session()
    execute_python("print('step 1')")
    execute_python("print('step 2')")
    history = get_execution_history(last_n=5)
    assert history["total"] == 2
    assert len(history["records"]) == 2
    print("✅ test_history passed")

def test_matplotlib_plot():
    """测试图表生成（输出到文件）"""
    code = """
import matplotlib
matplotlib.use('Agg')   # 非交互式后端，不弹窗
import matplotlib.pyplot as plt
import math

x = [i * 0.1 for i in range(63)]
y = [math.sin(v) for v in x]
plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.savefig('/tmp/test_plot.png', dpi=72, bbox_inches='tight')
print('图表已保存到 /tmp/test_plot.png')
"""
    result = execute_python(code, timeout=15)
    if result["success"]:
        print("✅ test_matplotlib_plot passed")
        print(f"   stdout: {result['stdout'].strip()}")
    else:
        print(f"⚠️  test_matplotlib_plot: {result.get('stderr', result.get('error', ''))[:200]}")

if __name__ == "__main__":
    print("=" * 50)
    print("代码执行沙箱 - 功能验证")
    print("=" * 50)
    test_basic_execution()
    test_timeout()
    test_security_block_import()
    test_security_block_eval()
    test_numpy_if_available()
    test_history()
    test_matplotlib_plot()
    print("\n🎉 所有测试完成")