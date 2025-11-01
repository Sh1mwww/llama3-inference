#!/usr/bin/env python3
"""
分析旧的 layer_timings_70B.csv 数据，修正单位问题
"""
import pandas as pd

# 读取旧数据
df = pd.read_csv("layer_timings_70B.csv")

print(f"加载了 {len(df)} 条记录")
print(f"列: {list(df.columns)}")

# 检查单位问题
if 'elapsed_ms' in df.columns:
    print("\n旧格式：使用 elapsed_ms")
    df['total_ms'] = df['elapsed_ms']

# 按 layer_idx 分组（看是否有重复计时）
layer_counts = df.groupby('layer_idx').size()
print(f"\n每个 layer_idx 的记录数:")
print(layer_counts.value_counts())

# 如果有 80 个不同的 layer，说明是旧格式（每层一条记录）
if len(layer_counts) == 80:
    print("\n✅ 确认：旧格式（80 层 × N token = 重复记录）")

    # 计算真实的每 token 时间
    # 假设：每 80 条记录对应同一个 token
    tokens = df.groupby('token_idx')['total_ms'].agg(['mean', 'std', 'count'])

    print(f"\n每个 token 的平均时间:")
    print(tokens.head(10))

    # 估算真实吞吐量
    decode_tokens = tokens[tokens.index >= 2017]
    if len(decode_tokens) > 0:
        avg_per_token = decode_tokens['mean'].mean()
        throughput = 1000 / avg_per_token  # tokens/s

        print(f"\n真实性能估算（Decode 阶段）:")
        print(f"  平均每 token 时间: {avg_per_token:.4f} ms")
        print(f"  真实吞吐量:       {throughput:.4f} tokens/s")
        print(f"                    ({throughput * 60:.2f} tokens/min)")

    # 查看 compute_ms（如果有）
    if 'compute_ms' in df.columns:
        compute_df = df[df['compute_ms'].notna()]
        if len(compute_df) > 0:
            print(f"\nCompute_ms 统计:")
            print(f"  平均: {compute_df['compute_ms'].mean():.2f}")
            print(f"  最小: {compute_df['compute_ms'].min():.2f}")
            print(f"  最大: {compute_df['compute_ms'].max():.2f}")

            # 检查是否是微秒
            avg_compute = compute_df['compute_ms'].mean()
            avg_total = df['total_ms'].mean()

            if avg_compute > avg_total:
                print(f"\n⚠️  单位问题！compute_ms ({avg_compute:.2f}) > total_ms ({avg_total:.2f})")
                print(f"     可能 compute_ms 是微秒（us），需要除以 1000")

                # 转换后的值
                print(f"\n修正后:")
                print(f"  compute_ms (修正): {avg_compute / 1000:.4f} ms")
                print(f"  占比:             {(avg_compute / 1000 / avg_total) * 100:.1f}%")

                # 真实瓶颈分析
                compute_ms_corrected = avg_compute / 1000
                weight_load_ms = avg_total - compute_ms_corrected

                print(f"\n瓶颈分析（修正后）:")
                print(f"  计算时间:   {compute_ms_corrected:.4f} ms ({(compute_ms_corrected / avg_total) * 100:.1f}%)")
                print(f"  权重加载:   {weight_load_ms:.4f} ms ({(weight_load_ms / avg_total) * 100:.1f}%)")

                if weight_load_ms > compute_ms_corrected:
                    ratio = weight_load_ms / compute_ms_corrected
                    print(f"  ⚠️  权重加载是瓶颈 ({ratio:.2f}x 慢于计算)")
                else:
                    print(f"  ✅ 计算是瓶颈")

else:
    print("\n新格式：layer_idx=-1 表示 token 级别")
