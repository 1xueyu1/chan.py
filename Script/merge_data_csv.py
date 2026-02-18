import os
import re


# ==================================================
#               🔧 配置区（只改这里）
# ==================================================

BASE_DIR = "./btc_data"     # 数据根目录

MODE = "monthly"            # "monthly" 或 "daily"
SYMBOL = "BTCUSDT"          # 交易对
INTERVAL = "1h"            # 周期: 1m/5m/15m/1h/4h/1d 等

AUTO_OUTPUT_NAME = True     # 是否自动生成输出文件名
CUSTOM_OUTPUT_FILE = None   # 若 AUTO_OUTPUT_NAME=False，则使用这个

# ==================================================


def build_data_dir():
    return os.path.join(BASE_DIR, MODE, INTERVAL)


def build_output_path():
    if not AUTO_OUTPUT_NAME and CUSTOM_OUTPUT_FILE:
        return CUSTOM_OUTPUT_FILE

    file_name = f"{SYMBOL}_{INTERVAL}.csv"
    return os.path.join(BASE_DIR, file_name)


def get_sorted_files(data_dir):
    """
    获取并排序符合规则的文件
    """
    pattern = re.compile(
        rf"{SYMBOL}-{INTERVAL}-.*\.csv"
    )

    files = [f for f in os.listdir(data_dir) if pattern.match(f)]
    files.sort()

    return files


def merge_files(data_dir, files, output_path):
    """
    核心合并逻辑：
    - 自动过滤表头
    - 自动过滤非法行
    - 自动去重
    - 自动排序
    """

    all_data = {}
    total_lines = 0
    kept_lines = 0

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        print(f"正在读取: {file_name}")

        with open(file_path, "r", encoding="utf-8") as infile:

            for line in infile:

                total_lines += 1
                line = line.strip()

                # 跳过空行
                if not line:
                    continue

                # 只保留以数字开头的行（时间戳）
                if not line[0].isdigit():
                    continue

                parts = line.split(",")

                if len(parts) < 6:
                    continue  # 防止异常行

                open_time = parts[0]

                # 去重（同一个 open_time 只保留一个）
                all_data[open_time] = line
                kept_lines += 1

    print("\n开始排序数据...")

    # 按时间戳排序
    sorted_times = sorted(all_data.keys(), key=lambda x: int(x))

    print("开始写入文件...")

    with open(output_path, "w", encoding="utf-8") as outfile:

        for ts in sorted_times:
            outfile.write(all_data[ts] + "\n")

    print("\n==============================")
    print(f"原始总行数: {total_lines}")
    print(f"有效数据行: {len(sorted_times)}")
    print(f"去重后保留: {len(sorted_times)}")
    print("==============================")


def main():
    data_dir = build_data_dir()

    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return

    files = get_sorted_files(data_dir)

    if not files:
        print("❌ 未找到匹配文件")
        return

    print("\n将按以下顺序合并文件:")
    for f in files:
        print(f)

    output_path = build_output_path()

    print(f"\n输出文件: {output_path}\n")

    merge_files(data_dir, files, output_path)

    print("\n✅ 合并完成！")


if __name__ == "__main__":
    main()