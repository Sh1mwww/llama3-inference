#!/usr/bin/env python3
"""
生成runtime manifest的脚本

用途:
1. 如果有shapes_meta.json，从它生成runtime_manifest.json
2. 如果没有shapes_meta.json，可以从原始checkpoint重新打包到SSD并生成manifest
3. 自动使用模型名命名输出文件

使用方法:
  # 情况1: 从checkpoint重新打包 (自动命名到/data1，推荐!)
  python generate_manifest.py from-checkpoint /home/roger/.llama/checkpoints/Llama3.1-8B/ /dev/nvme0n1p4
  # 将自动生成: /data1/llama3.1-8b.shapes_meta.json 和 /data1/llama3.1-8b.runtime_manifest.json

  # 情况2: 从shapes_meta.json生成manifest (系统重装后)
  python generate_manifest.py from-meta /data1/llama3.1-8b.shapes_meta.json
  # 将自动生成: /data1/llama3.1-8b.runtime_manifest.json

  # 情况3: 指定输出目录
  python generate_manifest.py from-checkpoint /path/to/checkpoint /dev/nvme0n1p4 --output-dir /home/roger/backups

  # 情况4: 完全自定义文件名
  python generate_manifest.py from-checkpoint /path/to/checkpoint /dev/nvme0n1p4 \
      --meta-out /home/roger/my-model.shapes_meta.json \
      --manifest-out /home/roger/my-model.runtime_manifest.json
"""

import sys
import os
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from llama3.weights_io_ssd_dram import (
    build_runtime_manifest,
    pack_any_to_raw,
    get_logical_block_size,
    O_DIRECT,
    O_LARGEFILE
)


def extract_model_name(ckpt_path: str) -> str:
    """
    从checkpoint路径中提取模型名字
    例如: /path/to/Llama3.1-8B/ -> llama3.1-8b
    """
    path = Path(ckpt_path)

    # 如果是目录，使用目录名
    if path.is_dir():
        name = path.name
    else:
        # 如果是文件，使用文件名（去掉扩展名）
        name = path.stem

    # 清理名字：转小写，去掉特殊字符
    name = name.lower()
    name = re.sub(r'[^a-z0-9.-]', '-', name)
    name = re.sub(r'-+', '-', name)  # 合并多个连字符
    name = name.strip('-')

    return name or "llama-model"


def generate_from_shapes_meta(shapes_meta_path: str, output_path: str = None) -> None:
    """
    从shapes_meta.json生成runtime_manifest.json
    这是系统重装后最常用的方法
    """
    if not Path(shapes_meta_path).exists():
        raise FileNotFoundError(f"❌ shapes_meta文件不存在: {shapes_meta_path}")

    # 自动生成输出路径（如果未指定）
    if output_path is None:
        # 从shapes_meta文件名提取模型名
        meta_filename = Path(shapes_meta_path).stem  # 去掉扩展名
        if meta_filename.endswith('.shapes_meta'):
            model_name = meta_filename[:-len('.shapes_meta')]
        else:
            model_name = meta_filename

        output_path = f"/data1/{model_name}.runtime_manifest.json"

    print(f"🔄 从shapes_meta生成manifest...")
    print(f"   输入: {shapes_meta_path}")
    print(f"   输出: {output_path}")

    # 调用现有的函数
    result = build_runtime_manifest(shapes_meta_path, output_path)

    print(f"✅ Manifest生成成功: {result}")

    # 显示摘要信息
    with open(result, 'r') as f:
        manifest = json.load(f)

    print(f"\n📊 Manifest摘要:")
    print(f"   设备: {manifest['raw_device']}")
    print(f"   块大小: {manifest['block_size']} bytes")
    print(f"   头部保留: {manifest['header_reserve']} bytes")
    print(f"   参数总数: {len(manifest['params'])}")

    # 统计各层参数
    layer_counts = {}
    resident_count = 0
    stream_count = 0

    for p in manifest['params']:
        layer = p['layer']
        policy = p['policy']

        if policy == 'resident':
            resident_count += 1
        elif policy == 'stream':
            stream_count += 1

        if layer >= 0:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    print(f"   常驻参数: {resident_count}")
    print(f"   流式参数: {stream_count}")
    if layer_counts:
        print(f"   层数: {len(layer_counts)} (layer {min(layer_counts.keys())}-{max(layer_counts.keys())})")


def generate_from_checkpoint(ckpt_path: str, raw_device: str,
                             meta_out: str = None,
                             manifest_out: str = None,
                             output_dir: str = None,
                             header_reserve: int = 4*1024*1024) -> None:
    """
    从checkpoint完全重新打包到raw设备并生成manifest
    这会覆盖raw设备上的现有数据！
    """
    print(f"⚠️  警告: 这将会覆盖 {raw_device} 上的现有数据!")
    confirm = input(f"确认要继续吗? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ 操作已取消")
        return

    # 提取模型名字
    model_name = extract_model_name(ckpt_path)
    print(f"\n🔄 从checkpoint打包到raw设备...")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   模型名字: {model_name}")
    print(f"   Raw设备: {raw_device}")
    print(f"   头部保留: {header_reserve} bytes")

    # 确定输出目录
    if output_dir is None:
        # 默认使用 /data1
        output_dir = "/data1"

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 自动生成文件名（如果未指定）
    if meta_out is None:
        meta_out = str(Path(output_dir) / f"{model_name}.shapes_meta.json")

    if manifest_out is None:
        manifest_out = str(Path(output_dir) / f"{model_name}.runtime_manifest.json")

    print(f"\n📝 输出文件:")
    print(f"   shapes_meta: {meta_out}")
    print(f"   manifest: {manifest_out}")

    # Step 1: 打包checkpoint到raw设备，生成shapes_meta.json
    print(f"\n📦 Step 1: 打包权重到raw设备...")
    shapes_meta_path = pack_any_to_raw(
        ckpt_path,
        raw_device,
        shapes_meta_out=meta_out,
        header_reserve_bytes=header_reserve
    )
    print(f"✅ shapes_meta生成: {shapes_meta_path}")

    # Step 2: 从shapes_meta生成runtime_manifest
    print(f"\n📝 Step 2: 生成runtime manifest...")
    build_runtime_manifest(shapes_meta_path, manifest_out)
    print(f"✅ runtime_manifest生成: {manifest_out}")

    print(f"\n🎉 完成! 两个文件已生成:")
    print(f"   1. {shapes_meta_path}")
    print(f"      (保存此文件用于以后重新生成manifest)")
    print(f"   2. {manifest_out}")
    print(f"      (每次系统启动时使用)")
    print(f"\n💡 备份建议:")
    print(f"   cp {shapes_meta_path} ~/backups/")
    print(f"   # 或提交到git: git add {shapes_meta_path}")


def generate_template(raw_device: str, output_path: str = None,
                     n_layers: int = 80,
                     model_name: str = "llama-model") -> None:
    """
    生成一个manifest模板文件
    用于测试或者手动修改
    """
    # 自动生成输出路径（如果未指定）
    if output_path is None:
        output_path = f"/data1/{model_name}.runtime_manifest.json"

    print(f"📝 生成manifest模板...")
    print(f"   设备: {raw_device}")
    print(f"   模型名: {model_name}")
    print(f"   层数: {n_layers}")
    print(f"   输出: {output_path}")

    # 获取设备的块大小
    try:
        fd = os.open(raw_device, os.O_RDONLY | O_DIRECT | O_LARGEFILE)
        block_size = get_logical_block_size(fd)
        os.close(fd)
    except Exception as e:
        print(f"⚠️  无法打开设备 {raw_device}: {e}")
        print(f"   使用默认块大小: 4096")
        block_size = 4096

    # 创建模板
    manifest = {
        "version": 1,
        "raw_device": raw_device,
        "block_size": block_size,
        "header_reserve": 4 * 1024 * 1024,  # 4MB
        "params": []
    }

    # 添加示例参数 (需要根据实际情况修改)
    print(f"   ⚠️  注意: 这只是一个模板，参数信息需要根据实际情况填写!")

    # 保存
    Path(output_path).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"✅ 模板已生成: {output_path}")
    print(f"\n💡 提示: 请手动编辑文件添加参数信息，或使用其他命令从实际数据生成")


def check_raw_device(device_path: str) -> None:
    """检查raw设备是否可访问"""
    print(f"\n🔍 检查设备: {device_path}")

    if not Path(device_path).exists():
        print(f"❌ 设备不存在: {device_path}")
        print(f"\n可用的块设备:")
        os.system("lsblk | grep -E 'nvme|sd'")
        sys.exit(1)

    try:
        fd = os.open(device_path, os.O_RDONLY | O_DIRECT | O_LARGEFILE)
        block_size = get_logical_block_size(fd)
        os.close(fd)
        print(f"✅ 设备可访问")
        print(f"   块大小: {block_size} bytes")
    except PermissionError:
        print(f"❌ 权限不足，请使用sudo运行")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 无法访问设备: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="生成LLaMA3推理系统的manifest文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='命令', required=True)

    # 子命令1: from-meta
    parser_meta = subparsers.add_parser(
        'from-meta',
        help='从shapes_meta.json生成runtime_manifest.json (系统重装后推荐)'
    )
    parser_meta.add_argument(
        'shapes_meta',
        type=str,
        help='shapes_meta.json文件路径'
    )
    parser_meta.add_argument(
        '--out',
        type=str,
        default=None,
        help='输出的manifest路径 (默认: /data1/<模型名>.runtime_manifest.json)'
    )
    parser_meta.add_argument(
        '--check-device',
        action='store_true',
        help='检查raw设备是否可访问'
    )

    # 子命令2: from-checkpoint
    parser_ckpt = subparsers.add_parser(
        'from-checkpoint',
        help='从checkpoint重新打包到raw设备并生成manifest (会覆盖现有数据!)'
    )
    parser_ckpt.add_argument(
        'checkpoint',
        type=str,
        help='Checkpoint文件或目录路径 (支持.pth文件或包含consolidated*.pth的目录)'
    )
    parser_ckpt.add_argument(
        'raw_device',
        type=str,
        help='Raw设备路径 (如 /dev/nvme0n1p4)'
    )
    parser_ckpt.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录 (默认: /data1, 文件名自动使用模型名)'
    )
    parser_ckpt.add_argument(
        '--meta-out',
        type=str,
        default=None,
        help='shapes_meta.json输出路径 (如指定则覆盖自动命名)'
    )
    parser_ckpt.add_argument(
        '--manifest-out',
        type=str,
        default=None,
        help='runtime_manifest.json输出路径 (如指定则覆盖自动命名)'
    )
    parser_ckpt.add_argument(
        '--header-reserve',
        type=int,
        default=4*1024*1024,
        help='头部保留空间 (bytes, 默认: 4MB)'
    )

    # 子命令3: template
    parser_template = subparsers.add_parser(
        'template',
        help='生成manifest模板 (用于测试或手动修改)'
    )
    parser_template.add_argument(
        '--raw-device',
        type=str,
        required=True,
        help='Raw设备路径'
    )
    parser_template.add_argument(
        '--out',
        type=str,
        default=None,
        help='输出路径 (默认: /data1/<模型名>.runtime_manifest.json)'
    )
    parser_template.add_argument(
        '--model-name',
        type=str,
        default='llama-model',
        help='模型名字 (用于文件命名, 默认: llama-model)'
    )
    parser_template.add_argument(
        '--layers',
        type=int,
        default=80,
        help='层数 (默认: 80)'
    )

    args = parser.parse_args()

    try:
        if args.command == 'from-meta':
            if args.check_device:
                # 读取meta文件获取设备路径
                with open(args.shapes_meta, 'r') as f:
                    meta = json.load(f)
                check_raw_device(meta['raw_device'])

            generate_from_shapes_meta(args.shapes_meta, args.out)

        elif args.command == 'from-checkpoint':
            check_raw_device(args.raw_device)
            generate_from_checkpoint(
                args.checkpoint,
                args.raw_device,
                meta_out=args.meta_out,
                manifest_out=args.manifest_out,
                output_dir=args.output_dir,
                header_reserve=args.header_reserve
            )

        elif args.command == 'template':
            generate_template(args.raw_device, args.out, n_layers=args.layers, model_name=args.model_name)

        print(f"\n✅ 操作完成!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
