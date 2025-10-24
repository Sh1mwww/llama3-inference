#!/usr/bin/env python3
"""
Checkpoint 诊断工具
检查 checkpoint 文件的完整性和参数一致性
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple

def check_checkpoint_integrity(ckpt_dir: str) -> Dict:
    """
    检查 checkpoint 的完整性

    Returns:
        dict: 诊断结果
    """
    ckpt_path = Path(ckpt_dir)
    result = {
        "checkpoint_dir": str(ckpt_path),
        "params_json_exists": False,
        "checkpoint_files_exist": False,
        "params_json_content": None,
        "checkpoint_vocab_size": None,
        "vocab_size_match": False,
        "issues": [],
        "recommendations": []
    }

    # 1. 检查 params.json
    params_file = ckpt_path / "params.json"
    if params_file.exists():
        result["params_json_exists"] = True
        try:
            params = json.loads(params_file.read_text())
            result["params_json_content"] = params
            print(f"✅ params.json 存在")
            print(f"   vocab_size: {params.get('vocab_size')}")
            print(f"   n_layers: {params.get('n_layers')}")
            print(f"   dim: {params.get('dim')}")
        except Exception as e:
            result["issues"].append(f"无法读取 params.json: {e}")
            print(f"❌ 无法读取 params.json: {e}")
    else:
        result["issues"].append("params.json 不存在")
        print(f"❌ params.json 不存在")
        return result

    # 2. 检查 checkpoint 文件
    ckpt_files = list(ckpt_path.glob("consolidated.*.pth"))
    if ckpt_files:
        result["checkpoint_files_exist"] = True
        print(f"\n✅ 找到 {len(ckpt_files)} 个 checkpoint 文件")

        # 加载第一个文件检查 vocab_size
        try:
            print(f"   加载 {ckpt_files[0].name} 检查参数...")
            ckpt = torch.load(ckpt_files[0], map_location='cpu', weights_only=True)

            # 检查 embedding 层
            embed_keys = [k for k in ckpt.keys() if 'embed' in k.lower()]
            output_keys = [k for k in ckpt.keys() if k == 'output.weight']

            if embed_keys:
                embed_key = embed_keys[0]
                embed_shape = ckpt[embed_key].shape
                checkpoint_vocab = embed_shape[0]
                result["checkpoint_vocab_size"] = checkpoint_vocab
                print(f"   {embed_key}: {embed_shape}")
                print(f"   Checkpoint vocab_size: {checkpoint_vocab}")

            if output_keys:
                output_shape = ckpt['output.weight'].shape
                print(f"   output.weight: {output_shape}")

            # 检查是否匹配
            params_vocab = result["params_json_content"].get("vocab_size")
            if result["checkpoint_vocab_size"] and params_vocab:
                if result["checkpoint_vocab_size"] == params_vocab:
                    result["vocab_size_match"] = True
                    print(f"\n✅ vocab_size 匹配: {params_vocab}")
                else:
                    result["vocab_size_match"] = False
                    result["issues"].append(
                        f"vocab_size 不匹配: params.json={params_vocab}, checkpoint={result['checkpoint_vocab_size']}"
                    )
                    print(f"\n❌ vocab_size 不匹配!")
                    print(f"   params.json: {params_vocab}")
                    print(f"   checkpoint: {result['checkpoint_vocab_size']}")

                    # 分析问题
                    if result["checkpoint_vocab_size"] == 16032:
                        result["recommendations"].append(
                            "Checkpoint vocab_size=16032 可能是:\n"
                            "  1. 旧版本的 Llama 模型\n"
                            "  2. 使用了不同的 tokenizer\n"
                            "  3. Checkpoint 文件被截断或损坏\n"
                            "  4. 这不是官方的 Llama 3.1-70B checkpoint"
                        )

                    if params_vocab == 128256:
                        result["recommendations"].append(
                            "params.json vocab_size=128256 是 Llama 3.1 的标准配置"
                        )

        except Exception as e:
            result["issues"].append(f"无法加载 checkpoint 文件: {e}")
            print(f"❌ 无法加载 checkpoint: {e}")
    else:
        result["issues"].append("未找到 checkpoint 文件 (consolidated.*.pth)")
        print(f"❌ 未找到 checkpoint 文件")

    # 3. 检查 tokenizer
    print(f"\n📝 检查 tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        tokenizer_vocab = tokenizer.vocab_size
        print(f"   Tokenizer vocab_size: {tokenizer_vocab}")

        if result["checkpoint_vocab_size"] and tokenizer_vocab != result["checkpoint_vocab_size"]:
            result["issues"].append(
                f"Tokenizer vocab_size ({tokenizer_vocab}) 与 checkpoint ({result['checkpoint_vocab_size']}) 不匹配"
            )
            print(f"   ⚠️  与 checkpoint 不匹配!")
    except Exception as e:
        print(f"   ⚠️  无法加载 tokenizer: {e}")

    return result


def generate_fix_script(result: Dict, output_path: str = "fix_vocab_mismatch.sh"):
    """
    生成修复脚本
    """
    if result["vocab_size_match"]:
        print(f"\n✅ 无需修复，vocab_size 已匹配")
        return

    checkpoint_vocab = result["checkpoint_vocab_size"]
    params_vocab = result["params_json_content"]["vocab_size"]

    script_content = f"""#!/bin/bash
# Checkpoint vocab_size 修复脚本
# 自动生成于 diagnose_checkpoint.py

set -e

CKPT_DIR="{result['checkpoint_dir']}"

echo "================================================"
echo "Llama3 Checkpoint vocab_size 修复"
echo "================================================"
echo ""
echo "检测到的问题:"
echo "  params.json vocab_size: {params_vocab}"
echo "  checkpoint vocab_size: {checkpoint_vocab}"
echo ""

# 选项1: 使用 checkpoint 的 vocab_size (推荐用于测试)
fix_option_1() {{
    echo "选项1: 修改 params.json 匹配 checkpoint ({checkpoint_vocab})"
    echo "  ⚠️  注意: 这只是临时方案，用于测试系统是否工作"
    echo "  ⚠️  正确的做法是获取完整的 checkpoint 文件"
    echo ""

    # 备份原文件
    cp "$CKPT_DIR/params.json" "$CKPT_DIR/params.json.bak"
    echo "✅ 已备份: params.json.bak"

    # 修改 vocab_size
    python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

params_path = Path("{result['checkpoint_dir']}/params.json")
params = json.loads(params_path.read_text())
params['vocab_size'] = {checkpoint_vocab}
params_path.write_text(json.dumps(params, indent=2))
print(f"✅ 已修改 params.json: vocab_size={checkpoint_vocab}")
PYTHON_SCRIPT

    echo ""
    echo "✅ 修复完成！现在可以重新生成 manifest:"
    echo ""
    echo "cd /home/roger/llama3-inference"
    echo "/home/roger/.pyenv/versions/inference_proj/bin/python generate_manifest.py from-meta /data1/llama3.1-70b.shapes_meta.json --out /data1/llama3.1-70b.runtime_manifest.json"
}}

# 选项2: 恢复备份
restore_backup() {{
    if [ -f "$CKPT_DIR/params.json.bak" ]; then
        cp "$CKPT_DIR/params.json.bak" "$CKPT_DIR/params.json"
        echo "✅ 已恢复备份"
    else
        echo "❌ 未找到备份文件"
    fi
}}

# 主菜单
echo "请选择操作:"
echo "  1) 修改 params.json 匹配 checkpoint (临时方案)"
echo "  2) 恢复 params.json 备份"
echo "  3) 退出"
echo ""
read -p "请输入选项 (1-3): " choice

case $choice in
    1)
        fix_option_1
        ;;
    2)
        restore_backup
        ;;
    3)
        echo "退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac
"""

    output = Path(output_path)
    output.write_text(script_content)
    output.chmod(0o755)

    print(f"\n✅ 已生成修复脚本: {output_path}")
    print(f"\n运行方法:")
    print(f"  ./{output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="检查 checkpoint 完整性")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        default="/home/roger/.llama/checkpoints/Llama3.1-70B",
        nargs='?',
        help="Checkpoint 目录路径"
    )
    parser.add_argument(
        "--generate-fix",
        action="store_true",
        help="生成修复脚本"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Llama3 Checkpoint 诊断工具")
    print("=" * 60)
    print()

    result = check_checkpoint_integrity(args.checkpoint_dir)

    # 打印总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    if result["issues"]:
        print("\n❌ 发现的问题:")
        for i, issue in enumerate(result["issues"], 1):
            print(f"  {i}. {issue}")

    if result["recommendations"]:
        print("\n💡 建议:")
        for rec in result["recommendations"]:
            print(f"\n{rec}")

    if not result["issues"]:
        print("\n✅ 所有检查通过!")

    # 生成修复脚本
    if args.generate_fix or not result["vocab_size_match"]:
        if not result["vocab_size_match"]:
            generate_fix_script(result)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
