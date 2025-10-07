#!/usr/bin/env python3
"""
查找和诊断权重文件及manifest的工具脚本
"""

import os
import sys
import json
from pathlib import Path


def scan_for_files():
    """扫描系统查找相关文件"""
    print("🔍 扫描系统中的权重和manifest文件...\n")

    # 搜索路径
    search_paths = [
        "/data1",
        "/data2",
        "/home/roger",
        "/mnt",
        "/tmp",
        "/dev/shm"
    ]

    findings = {
        'shapes_meta': [],
        'runtime_manifest': [],
        'checkpoints': [],
        'raw_devices': []
    }

    # 查找JSON文件
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue

        try:
            for root, dirs, files in os.walk(base_path, followlinks=False):
                # 跳过一些不相关的目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]

                for file in files:
                    if file.endswith('.shapes_meta.json'):
                        findings['shapes_meta'].append(os.path.join(root, file))
                    elif 'runtime_manifest' in file and file.endswith('.json'):
                        findings['runtime_manifest'].append(os.path.join(root, file))
                    elif file.endswith('.pth') and 'consolidated' in file:
                        findings['checkpoints'].append(os.path.join(root, file))

        except (PermissionError, OSError):
            continue

    # 查找块设备
    print("📀 块设备信息:")
    os.system("lsblk | grep -E 'nvme|sd'")
    print()

    # 显示发现的文件
    print("=" * 60)
    print("📁 发现的文件:\n")

    if findings['shapes_meta']:
        print("✅ shapes_meta.json 文件:")
        for f in findings['shapes_meta']:
            size = os.path.getsize(f) / 1024
            print(f"   - {f} ({size:.1f} KB)")
    else:
        print("❌ 未找到 shapes_meta.json 文件")

    print()

    if findings['runtime_manifest']:
        print("✅ runtime_manifest.json 文件:")
        for f in findings['runtime_manifest']:
            size = os.path.getsize(f) / 1024
            print(f"   - {f} ({size:.1f} KB)")
    else:
        print("❌ 未找到 runtime_manifest.json 文件")

    print()

    if findings['checkpoints']:
        print(f"✅ 找到 {len(findings['checkpoints'])} 个checkpoint文件 (仅显示前5个):")
        for f in findings['checkpoints'][:5]:
            size = os.path.getsize(f) / (1024**3)
            print(f"   - {f} ({size:.1f} GB)")
    else:
        print("❌ 未找到 checkpoint (.pth) 文件")

    print()
    print("=" * 60)

    return findings


def check_manifest(manifest_path: str):
    """检查manifest文件的内容和有效性"""
    print(f"\n📋 检查manifest: {manifest_path}\n")

    if not os.path.exists(manifest_path):
        print(f"❌ 文件不存在: {manifest_path}")
        return

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # 基本信息
        print("📊 基本信息:")
        print(f"   版本: {manifest.get('version', 'N/A')}")
        print(f"   Raw设备: {manifest.get('raw_device', 'N/A')}")
        print(f"   块大小: {manifest.get('block_size', 'N/A')} bytes")
        print(f"   头部保留: {manifest.get('header_reserve', 'N/A')} bytes")

        # 检查设备是否存在
        raw_dev = manifest.get('raw_device')
        if raw_dev:
            if os.path.exists(raw_dev):
                print(f"   ✅ 设备存在: {raw_dev}")
            else:
                print(f"   ❌ 设备不存在: {raw_dev}")

        # 参数统计
        params = manifest.get('params', [])
        print(f"\n📦 参数统计:")
        print(f"   总参数数: {len(params)}")

        if params:
            # 按策略统计
            resident = sum(1 for p in params if p.get('policy') == 'resident')
            stream = sum(1 for p in params if p.get('policy') == 'stream')
            print(f"   常驻参数: {resident}")
            print(f"   流式参数: {stream}")

            # 按层统计
            layers = set(p.get('layer', -1) for p in params if p.get('layer', -1) >= 0)
            if layers:
                print(f"   层数范围: {min(layers)} - {max(layers)} (共 {len(layers)} 层)")

            # 总大小
            total_bytes = sum(p.get('nbytes', 0) for p in params)
            print(f"   总大小: {total_bytes / (1024**3):.2f} GB")

            # 示例参数
            print(f"\n📝 前5个参数示例:")
            for i, p in enumerate(params[:5]):
                name = p.get('name', 'N/A')
                shape = p.get('shape', [])
                dtype = p.get('dtype', 'N/A')
                policy = p.get('policy', 'N/A')
                print(f"   {i+1}. {name}")
                print(f"      shape={shape}, dtype={dtype}, policy={policy}")

        print(f"\n✅ Manifest文件有效")

    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")


def show_usage():
    """显示使用说明"""
    print("\n" + "="*60)
    print("💡 下一步操作建议:\n")

    print("1️⃣ 如果找到了 shapes_meta.json:")
    print("   python generate_manifest.py from-meta /path/to/shapes_meta.json \\\n"
          "          --out /data1/runtime_manifest.json\n")

    print("2️⃣ 如果没找到 shapes_meta.json，但有checkpoint:")
    print("   python generate_manifest.py from-checkpoint /path/to/checkpoint \\\n"
          "          /dev/nvme0n1p4 --meta-out /data1/llama.shapes_meta.json\n")

    print("3️⃣ 检查现有的manifest文件:")
    print("   python find_weights.py check /path/to/manifest.json\n")

    print("4️⃣ 如果权重已经在SSD上，但没有任何JSON文件:")
    print("   需要找到原始的shapes_meta.json备份，或者重新打包checkpoint\n")

    print("="*60)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        if len(sys.argv) < 3:
            print("用法: python find_weights.py check <manifest.json>")
            sys.exit(1)
        check_manifest(sys.argv[2])
    else:
        findings = scan_for_files()
        show_usage()

        # 如果找到了shapes_meta，直接提供命令
        if findings['shapes_meta']:
            print(f"\n🎯 快速命令 (使用找到的第一个shapes_meta):")
            meta_path = findings['shapes_meta'][0]
            print(f"   python generate_manifest.py from-meta {meta_path}")


if __name__ == '__main__':
    main()
