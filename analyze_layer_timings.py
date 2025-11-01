#!/usr/bin/env python3
"""
分析 layer_timings_70B.csv 的工具脚本
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_timings(csv_path="layer_timings_70B.csv"):
    """分析层级时间数据"""
    if not Path(csv_path).exists():
        print(f"❌ 文件不存在: {csv_path}")
        return

    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"\n📊 加载了 {len(df)} 条记录")
    print(f"列: {list(df.columns)}")

    # 兼容旧格式：如果有 elapsed_ms 但没有 total_ms，重命名
    if 'elapsed_ms' in df.columns and 'total_ms' not in df.columns:
        df['total_ms'] = df['elapsed_ms']

    # 检查数据格式（新格式：layer_idx=-1 表示整个 token）
    is_token_level = (df['layer_idx'] == -1).any()
    if is_token_level:
        print("✅ 检测到 Token 级别计时（每个 token 一条记录）")
        # 只保留 token 级别的记录
        df = df[df['layer_idx'] == -1].copy()
    else:
        print("⚠️  检测到旧格式（每层一条记录，可能有重复计时问题）")

    # 检查是否有详细时间分解
    has_detailed_timing = 'compute_ms' in df.columns and 'weight_load_ms' in df.columns

    # ★ 修正旧数据的单位问题：如果 compute_ms 异常大（可能是微秒），自动转换
    if has_detailed_timing and 'compute_ms' in df.columns:
        compute_df = df[df['compute_ms'].notna()]
        if len(compute_df) > 0:
            avg_compute = compute_df['compute_ms'].mean()
            avg_total = df['total_ms'].mean()

            # 如果 compute_ms 平均值 > total_ms，说明单位可能是 us
            if avg_compute > avg_total * 10:  # 明显异常
                print("⚠️  检测到旧数据单位问题（compute_ms 实际是 us），自动转换为 ms")
                df['compute_ms'] = df['compute_ms'] / 1000.0
                df['weight_load_ms'] = df['weight_load_ms'] / 1000.0

    # 基础统计
    print("\n" + "=" * 70)
    print("整体统计 (Total Time)")
    print("=" * 70)
    print(f"总时间:       {df['total_ms'].sum():.2f} ms")
    print(f"平均时间:     {df['total_ms'].mean():.4f} ms")
    print(f"中位数:       {df['total_ms'].median():.4f} ms")
    print(f"标准差:       {df['total_ms'].std():.4f} ms")
    print(f"最小值:       {df['total_ms'].min():.4f} ms")
    print(f"最大值:       {df['total_ms'].max():.4f} ms")

    # ★ 新增：时间分解统计
    if has_detailed_timing:
        print("\n" + "=" * 70)
        print("时间分解统计 (Compute vs Weight Load)")
        print("=" * 70)
        compute_df = df[df['compute_ms'].notna()]
        weight_df = df[df['weight_load_ms'].notna()]

        if len(compute_df) > 0:
            print(f"\n计算时间 (Compute):")
            print(f"  平均: {compute_df['compute_ms'].mean():.4f} ms")
            print(f"  占比: {compute_df['compute_ms'].mean() / df['total_ms'].mean() * 100:.1f}%")

        if len(weight_df) > 0:
            print(f"\n权重加载时间 (Weight Load):")
            print(f"  平均: {weight_df['weight_load_ms'].mean():.4f} ms")
            print(f"  占比: {weight_df['weight_load_ms'].mean() / df['total_ms'].mean() * 100:.1f}%")

        if len(compute_df) > 0 and len(weight_df) > 0:
            print(f"\n瓶颈分析:")
            compute_avg = compute_df['compute_ms'].mean()
            weight_avg = weight_df['weight_load_ms'].mean()
            if weight_avg > compute_avg:
                ratio = weight_avg / compute_avg
                print(f"  ⚠️  权重加载是瓶颈 ({ratio:.2f}x 慢于计算)")
                print(f"      建议：优化 SSD → CPU → GPU 流水线")
            else:
                ratio = compute_avg / weight_avg
                print(f"  ✅ 计算是瓶颈 ({ratio:.2f}x 慢于权重加载)")
                print(f"      GPU 利用率良好")

    # 按层统计（仅在旧格式时有意义）
    if not is_token_level and 'layer_idx' in df.columns:
        print("\n" + "=" * 70)
        print("按层统计 (Top 10 最慢)")
        print("=" * 70)
        layer_stats = df.groupby('layer_idx')['total_ms'].agg(['mean', 'std', 'count'])
        layer_stats = layer_stats.sort_values('mean', ascending=False)
        print(layer_stats.head(10).to_string())

    # 按 token 位置统计（prefill vs decode）
    print("\n" + "=" * 70)
    print("按 token 位置统计")
    print("=" * 70)
    token_stats = df.groupby('token_idx')['total_ms'].agg(['mean', 'std', 'count'])
    print(f"Prefill 阶段 (token 0-10):")
    print(token_stats.head(10).to_string())

    if len(token_stats) > 10:
        print(f"\nDecode 阶段 (token 10+):")
        print(token_stats.tail(10).to_string())

    # 找出异常慢的记录
    print("\n" + "=" * 70)
    print("异常慢的记录 (> mean + 2*std)")
    print("=" * 70)
    threshold = df['total_ms'].mean() + 2 * df['total_ms'].std()
    outliers = df[df['total_ms'] > threshold]
    if len(outliers) > 0:
        print(f"找到 {len(outliers)} 条异常记录:")
        cols_to_show = ['batch_idx', 'layer_idx', 'token_idx', 'total_ms']
        if has_detailed_timing:
            cols_to_show.extend(['compute_ms', 'weight_load_ms'])
        print(outliers[cols_to_show].to_string())
    else:
        print("没有异常慢的记录")

    # Prefill vs Decode 对比
    print("\n" + "=" * 70)
    print("Prefill vs Decode 对比")
    print("=" * 70)
    # 假设 token_idx < 10 是 prefill，>= 10 是 decode
    prefill = df[df['token_idx'] < 10]
    decode = df[df['token_idx'] >= 10]

    if len(prefill) > 0 and len(decode) > 0:
        print(f"Prefill: {prefill['total_ms'].mean():.4f} ms (avg), {len(prefill)} records")
        print(f"Decode:  {decode['total_ms'].mean():.4f} ms (avg), {len(decode)} records")
        print(f"Speedup: {prefill['total_ms'].mean() / decode['total_ms'].mean():.2f}x")

    # ===== Token 级别分析 =====
    if is_token_level:
        print("\n" + "=" * 70)
        print("Token 生成时间趋势")
        print("=" * 70)

        # 前 10 个 token vs 后 10 个 token
        if len(df) >= 20:
            first_10 = df.nsmallest(10, 'token_idx')
            last_10 = df.nlargest(10, 'token_idx')

            print(f"前 10 个 token 平均时间: {first_10['total_ms'].mean():.4f} ms")
            print(f"后 10 个 token 平均时间: {last_10['total_ms'].mean():.4f} ms")

            if last_10['total_ms'].mean() > first_10['total_ms'].mean():
                print(f"⚠️  后期变慢: {(last_10['total_ms'].mean() / first_10['total_ms'].mean() - 1) * 100:.1f}%")
            else:
                print(f"✅ 后期加速: {(1 - last_10['total_ms'].mean() / first_10['total_ms'].mean()) * 100:.1f}%")
    else:
        # 旧格式：Layer 0 分析
        print("\n" + "=" * 70)
        print("Layer 0 性能分析（冷启动效应）")
        print("=" * 70)
        layer0 = df[df['layer_idx'] == 0]
        if len(layer0) > 0:
            print(f"Layer 0 平均: {layer0['total_ms'].mean():.4f} ms")
            print(f"所有层平均:   {df['total_ms'].mean():.4f} ms")
            print(f"Layer 0 慢:   {(layer0['total_ms'].mean() / df['total_ms'].mean() - 1) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Decode 阶段稳定性（排除 prefill 和 layer 0）")
    print("=" * 70)
    # 只看 decode 阶段（token >= 2017）且非 layer 0
    stable_decode = df[(df['token_idx'] >= 2017) & (df['layer_idx'] > 0)]
    if len(stable_decode) > 0:
        print(f"稳定 decode 记录数: {len(stable_decode)}")
        print(f"平均时间:           {stable_decode['total_ms'].mean():.4f} ms")
        print(f"标准差:             {stable_decode['total_ms'].std():.4f} ms")
        print(f"变异系数 (CV):      {stable_decode['total_ms'].std() / stable_decode['total_ms'].mean() * 100:.2f}%")
        print(f"最小值:             {stable_decode['total_ms'].min():.4f} ms")
        print(f"最大值:             {stable_decode['total_ms'].max():.4f} ms")

    print("\n" + "=" * 70)
    print("吞吐量估算")
    print("=" * 70)
    if len(decode) > 0:
        if is_token_level:
            # 新格式：直接使用 token 时间
            avg_time_per_token = decode['total_ms'].mean() / 1000  # 转换为秒
            tokens_per_sec = 1 / avg_time_per_token if avg_time_per_token > 0 else 0

            print(f"平均每 token 时间:  {decode['total_ms'].mean():.4f} ms")
            print(f"✅ 真实吞吐量:       {tokens_per_sec:.4f} tokens/s")
            print(f"                    ({tokens_per_sec * 60:.2f} tokens/min)")

            if len(prefill) > 0:
                print(f"Prefill 延迟:       {prefill['total_ms'].mean():.2f} ms（首个 token）")
        else:
            # 旧格式：警告可能不准确
            avg_time_per_layer = decode['total_ms'].mean()
            n_layers = 80
            total_time_per_token = avg_time_per_layer * n_layers / 1000  # 转换为秒
            tokens_per_sec = 1 / total_time_per_token

            print(f"⚠️  警告：旧格式数据，可能不准确")
            print(f"每层平均时间:       {avg_time_per_layer:.4f} ms")
            print(f"总层数:             {n_layers}")
            print(f"每 token 总时间:    {total_time_per_token:.4f} s")
            print(f"逐层累加吞吐量:     {tokens_per_sec:.4f} tokens/s")
            print(f"    (注意：这可能不是真实吞吐量)")
            print(f"延迟 (TTFT):        ~{prefill['total_ms'].mean() * n_layers / 1000:.2f} s（首个 token）")

    print("\n" + "=" * 70)
    print("性能瓶颈诊断")
    print("=" * 70)

    if is_token_level:
        # Token 级别：分析时间抖动
        std_ratio = df['total_ms'].std() / df['total_ms'].mean() * 100
        print(f"时间变异系数 (CV): {std_ratio:.2f}%")

        if std_ratio > 20:
            print(f"⚠️  时间抖动较大 (CV > 20%)")
            print(f"    可能原因：权重加载/KV I/O 竞争")
        else:
            print(f"✅ 时间稳定 (CV < 20%)")

        # 查找异常慢的 token
        threshold = df['total_ms'].mean() + 2 * df['total_ms'].std()
        slow_tokens = df[df['total_ms'] > threshold]
        if len(slow_tokens) > 0:
            print(f"\n⚠️  异常慢的 token（> mean + 2σ）: {len(slow_tokens)} 个")
            print(f"   Token 位置: {slow_tokens['token_idx'].tolist()[:10]}")
    else:
        # 旧格式：按层分析
        layer_avg = df.groupby('layer_idx')['total_ms'].mean()
        overall_avg = df['total_ms'].mean()
        slow_layers = layer_avg[layer_avg > overall_avg * 1.05]  # 慢 5% 以上

        if len(slow_layers) > 0:
            print(f"⚠️  持续慢的层（> 平均值 5%）：")
            for lid, avg_time in slow_layers.items():
                print(f"  Layer {lid:2d}: {avg_time:.4f} ms ({(avg_time / overall_avg - 1) * 100:.1f}% 慢)")
        else:
            print("✅ 所有层性能均衡，无明显瓶颈层")

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "layer_timings_70B.csv"
    analyze_timings(csv_file)
