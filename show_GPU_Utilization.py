import subprocess
import time
import sys


def monitor_gpu():
    try:
        while True:
            # 调用 nvidia-smi 命令获取 GPU 信息
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.total,memory.used,memory.free',
                                     '--format=csv,noheader,nounits'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 解析结果
            gpu_info = result.stdout.strip()
            if gpu_info:
                gpu_utilization, memory_total, memory_used, memory_free = gpu_info.split(', ')

                # 清除上一行输出
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")  # 清除当前行
                print(f"\rGPU Utilization: {gpu_utilization: <3}%", end="")

            else:
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")  # 清除当前行
                print("No GPU detected or nvidia-smi failed.")
                break

            time.sleep(1)  # 每秒更新一次
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")


if __name__ == "__main__":
    monitor_gpu()
