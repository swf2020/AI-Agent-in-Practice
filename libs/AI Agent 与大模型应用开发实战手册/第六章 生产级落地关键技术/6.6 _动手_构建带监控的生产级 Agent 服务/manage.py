#!/usr/bin/env python3
"""
服务管理脚本 - 统一管理 FastAPI 和 ARQ Worker 的启动、停止和重启

使用方式:
    python manage.py start      # 启动所有服务
    python manage.py stop       # 停止所有服务
    python manage.py restart    # 重启所有服务
    python manage.py status     # 查看服务状态
"""

import os
import sys
import signal
import subprocess
import time
import argparse
from typing import Optional

# 配置参数
FASTAPI_PORT = 8000
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_BIN = "/opt/homebrew/anaconda3/bin/python"

# 进程 ID 文件路径
PID_FILES = {
    "fastapi": os.path.join(WORK_DIR, "fastapi.pid"),
    "worker": os.path.join(WORK_DIR, "worker.pid"),
}


def get_pid(pid_file: str) -> Optional[int]:
    """从 pid 文件读取进程 ID"""
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None
    return None


def is_process_running(pid: int) -> bool:
    """检查进程是否运行"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_process(pid_file: str, service_name: str) -> bool:
    """停止指定服务"""
    pid = get_pid(pid_file)
    if pid and is_process_running(pid):
        print(f"⏹️  正在停止 {service_name} (PID: {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            # 等待进程退出
            for _ in range(10):
                if not is_process_running(pid):
                    break
                time.sleep(0.5)
            if os.path.exists(pid_file):
                os.remove(pid_file)
            print(f"   ✅ {service_name} 已停止")
            return True
        except OSError as e:
            print(f"   ❌ 停止 {service_name} 失败: {e}")
            return False
    else:
        print(f"   ℹ️  {service_name} 未运行")
        if os.path.exists(pid_file):
            os.remove(pid_file)
        return True


def start_fastapi() -> bool:
    """启动 FastAPI 服务"""
    pid_file = PID_FILES["fastapi"]
    
    # 检查是否已运行
    pid = get_pid(pid_file)
    if pid and is_process_running(pid):
        print(f"   ⚠️  FastAPI 已在运行 (PID: {pid})")
        return True
    
    print("🚀 启动 FastAPI 服务...")
    env = os.environ.copy()
    env["PYTHONPATH"] = WORK_DIR
    
    try:
        # 使用 nohup 后台运行
        cmd = [
            "nohup",
            PYTHON_BIN, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", str(FASTAPI_PORT),
            "--reload"
        ]
        
        proc = subprocess.Popen(
            cmd,
            cwd=WORK_DIR,
            env=env,
            stdout=open(os.path.join(WORK_DIR, "fastapi.log"), "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        
        # 保存 PID
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))
        
        # 等待启动
        time.sleep(3)
        
        # 检查是否启动成功
        if is_process_running(proc.pid):
            print(f"   ✅ FastAPI 启动成功 (PID: {proc.pid}, 端口: {FASTAPI_PORT})")
            return True
        else:
            print("   ❌ FastAPI 启动失败，请查看 fastapi.log")
            if os.path.exists(pid_file):
                os.remove(pid_file)
            return False
            
    except Exception as e:
        print(f"   ❌ 启动 FastAPI 失败: {e}")
        return False


def start_worker() -> bool:
    """启动 ARQ Worker"""
    pid_file = PID_FILES["worker"]
    
    # 检查是否已运行
    pid = get_pid(pid_file)
    if pid and is_process_running(pid):
        print(f"   ⚠️  ARQ Worker 已在运行 (PID: {pid})")
        return True
    
    print("🚀 启动 ARQ Worker...")
    env = os.environ.copy()
    env["PYTHONPATH"] = WORK_DIR
    
    try:
        # 使用 nohup 后台运行
        cmd = [
            "nohup",
            PYTHON_BIN, "-m", "arq", "worker.WorkerSettings"
        ]
        
        proc = subprocess.Popen(
            cmd,
            cwd=WORK_DIR,
            env=env,
            stdout=open(os.path.join(WORK_DIR, "worker.log"), "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        
        # 保存 PID
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))
        
        # 等待启动
        time.sleep(2)
        
        # 检查是否启动成功
        if is_process_running(proc.pid):
            print(f"   ✅ ARQ Worker 启动成功 (PID: {proc.pid})")
            return True
        else:
            print("   ❌ ARQ Worker 启动失败，请查看 worker.log")
            if os.path.exists(pid_file):
                os.remove(pid_file)
            return False
            
    except Exception as e:
        print(f"   ❌ 启动 ARQ Worker 失败: {e}")
        return False


def check_health() -> bool:
    """检查服务健康状态"""
    try:
        import httpx
        response = httpx.get(f"http://localhost:{FASTAPI_PORT}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   🩺 健康检查通过: {data}")
            return True
        else:
            print(f"   ❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 健康检查失败: {e}")
        return False


def status():
    """查看服务状态"""
    print("📊 服务状态检查:")
    
    # FastAPI 状态
    pid = get_pid(PID_FILES["fastapi"])
    if pid and is_process_running(pid):
        print(f"   FastAPI: 🟢 运行中 (PID: {pid}, 端口: {FASTAPI_PORT})")
    else:
        print("   FastAPI: 🔴 未运行")
        if os.path.exists(PID_FILES["fastapi"]):
            os.remove(PID_FILES["fastapi"])
    
    # ARQ Worker 状态
    pid = get_pid(PID_FILES["worker"])
    if pid and is_process_running(pid):
        print(f"   ARQ Worker: 🟢 运行中 (PID: {pid})")
    else:
        print("   ARQ Worker: 🔴 未运行")
        if os.path.exists(PID_FILES["worker"]):
            os.remove(PID_FILES["worker"])


def stop():
    """停止所有服务"""
    print("⏹️  停止所有服务:")
    stop_process(PID_FILES["fastapi"], "FastAPI")
    stop_process(PID_FILES["worker"], "ARQ Worker")


def start():
    """启动所有服务"""
    print("🚀 启动所有服务:")
    
    # 确保环境变量正确加载
    env_file = os.path.join(WORK_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value.strip('"').strip("'")
    
    # 启动服务
    success = True
    success &= start_fastapi()
    success &= start_worker()
    
    # 健康检查
    if success:
        print("\n🩺 健康检查:")
        check_health()
    
    print("\n📋 服务启动完成！")
    print(f"   API 文档: http://localhost:{FASTAPI_PORT}/docs")
    print(f"   健康检查: http://localhost:{FASTAPI_PORT}/health")
    print(f"   指标接口: http://localhost:{FASTAPI_PORT}/metrics")


def restart():
    """重启所有服务"""
    print("🔄 重启所有服务:")
    print("-" * 50)
    
    # 先停止
    stop()
    print()
    
    # 等待清理
    time.sleep(1)
    
    # 再启动
    start()


def main():
    parser = argparse.ArgumentParser(description="服务管理脚本")
    parser.add_argument("command", choices=["start", "stop", "restart", "status"],
                        help="操作命令: start|stop|restart|status")
    args = parser.parse_args()
    
    # 确保在正确的工作目录
    os.chdir(WORK_DIR)
    
    if args.command == "start":
        start()
    elif args.command == "stop":
        stop()
    elif args.command == "restart":
        restart()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
