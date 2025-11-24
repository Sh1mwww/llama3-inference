#!/usr/bin/env python3
"""
检查 staging buffer 的读取范围
"""

import json
from llama3.weight_lbt import classify_group

manifest_path = "/data1/70b-fixed.runtime_manifest.json"

with open(manifest_path, "r") as f:
    manifest = json.load(f)

layer_id = 0
layer_params = [
    p for p in manifest.get("params", [])
    if int(p.get("layer", -1)) == layer_id and p.get("policy") == "stream"
]

# 按 MHA/FFN 分组
groups = {"mha": [], "ffn": []}
for p in layer_params:
    g = classify_group(p["name"])
    if g in ("mha", "ffn"):
        groups[g].append(p)

print("=" * 80)
print("MHA 组 staging buffer 布局")
print("=" * 80)

gparams = sorted(groups["mha"], key=lambda x: int(x["offset"]))
start_offset = int(gparams[0]["offset"])
end_offset = max(int(p["offset"]) + int(p["stride"]) for p in gparams)
span = end_offset - start_offset

print(f"Start offset: {start_offset}")
print(f"End offset:   {end_offset}")
print(f"Span:         {span} bytes ({span/(1024**2):.2f} MB)\n")

print("参数在 staging buffer 中的位置:")
for pinfo in gparams:
    param_off = int(pinfo["offset"]) - start_offset
    nbytes = int(pinfo["nbytes"])
    stride = int(pinfo["stride"])
    name = pinfo["name"]

    print(f"\n{name}:")
    print(f"  File offset:    {pinfo['offset']:>12} bytes")
    print(f"  Staging offset: {param_off:>12} bytes ({param_off/(1024**2):>8.2f} MB)")
    print(f"  nbytes:         {nbytes:>12} bytes ({nbytes/(1024**2):>8.2f} MB)")
    print(f"  stride:         {stride:>12} bytes ({stride/(1024**2):>8.2f} MB)")
    print(f"  Staging range:  [{param_off}, {param_off + nbytes})")

    # 检查范围是否在 span 内
    if param_off + nbytes > span:
        print(f"  ⚠️  ERROR: 超出 staging buffer 范围！")
