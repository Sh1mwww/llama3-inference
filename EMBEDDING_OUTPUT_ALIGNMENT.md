# Embedding 与 Output 层维度对齐修复

## 问题描述

在 Transformer 模型中，**输入 embedding 层和输出投影层必须在 vocab 维度上对齐**：

1. **embed_tokens (输入)**：`nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)`
   - 权重形状：`[vocab_size, dim]`
   - 功能：token ID → embedding vector

2. **output/lm_head (输出)**：`nn.Linear(in_features=dim, out_features=vocab_size)`
   - 权重形状：`[vocab_size, dim]`（注意：Linear 权重是转置的）
   - 功能：hidden states → vocab logits

**如果 vocab_size 不一致**：
- 前向传播时，embedding 查表可能越界
- 输出 logits 的维度与实际词表大小不匹配
- 权重加载时会出现形状不匹配错误

## 当前状态验证

### Llama 3.1 70B 模型：

```
vocab_size: 128256
dim: 8192

embed_tokens.weight:  [128256, 8192]
output.weight:        [128256, 8192]
```

✅ **已对齐**：两者的 vocab_size 都是 128256

## 代码修复

### 1. 模型初始化验证 (llama3/model.py:24-51)

```python
# ★ 确保 vocab_size 合法（必须 > 0）
assert args.vocab_size > 0, f"vocab_size 必须 > 0，当前值: {args.vocab_size}"

# 创建 embedding 层：num_embeddings = vocab_size，embedding_dim = dim
# 注意：num_embeddings 必须等于后续加载权重的 shape[0]，否则前向查表会出错
self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)

# ... [transformer layers] ...

# ★ 输出投影层：必须与 embed_tokens 在 vocab 维度对齐
# nn.Linear(in_features=dim, out_features=vocab_size)
# output.weight.shape = [vocab_size, dim]，与 embed_tokens.weight 同形状
# 许多模型会 tie weights（共享权重），但即使不共享，vocab_size 也必须一致
self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
```

### 2. 权重加载时形状验证 (llama3/weights_io_ssd_dram.py:458-483)

增强错误消息，区分 embedding 和 output 层的错误：

```python
# ★ 形状验证：确保加载的权重与模块参数形状一致
# 特别关键的对齐要求：
# 1. nn.Embedding(num_embeddings, embedding_dim): num_embeddings 必须等于 weight.shape[0]
# 2. output/lm_head: nn.Linear(dim, vocab_size): vocab_size 必须等于 weight.shape[0]
if tuple(p["shape"]) != tuple(param.shape):
    # 给出详细的错误信息，包括 embed_tokens 和 output 的特定提示
    raise RuntimeError(...)
```

## 测试验证

运行测试脚本：

```bash
python3 test_output_embedding_alignment.py
```

### 测试结果：

```
✅ vocab 维度对齐:
   embed.num_embeddings (128256) == output.out_features (128256)

✅ hidden 维度对齐:
   embed.embedding_dim (8192) == output.in_features (8192)

✅ Manifest 形状验证通过:
   embed_tokens: (128256, 8192)
   output:       (128256, 8192)

✅ 前向传播路径验证:
   tokens [2, 5] → embeddings [2, 5, 8192] → logits [2, 5, 128256]
```

## 数据流示意

```
Input tokens:          [batch, seq_len]              例: [2, 5]
       ↓
embed_tokens:          [batch, seq_len, dim]         例: [2, 5, 8192]
  (查表 vocab_size=128256)
       ↓
Transformer layers:    [batch, seq_len, dim]         例: [2, 5, 8192]
       ↓
output (Linear):       [batch, seq_len, vocab_size]  例: [2, 5, 128256]
  (投影到 vocab_size=128256)
       ↓
logits over vocab:     [batch, seq_len, vocab_size]  例: [2, 5, 128256]
```

## 关键要点

1. **vocab_size 来源**：从 `params.json` 读取，必须与 checkpoint 一致
2. **embed_tokens**：`num_embeddings` 必须等于 `vocab_size`
3. **output**：`out_features` 必须等于 `vocab_size`
4. **形状对齐**：`embed_tokens.weight.shape[0] == output.weight.shape[0] == vocab_size`
5. **早期检测**：在权重加载时验证形状，避免运行时错误

## 相关文件

- [llama3/model.py](llama3/model.py) - 模型定义，添加了注释和验证
- [llama3/weights_io_ssd_dram.py](llama3/weights_io_ssd_dram.py) - 权重加载，增强了错误提示
- [test_output_embedding_alignment.py](test_output_embedding_alignment.py) - 对齐验证测试

---

## ⚠️ 重要：分片权重与完整权重

### 分片权重（Vocab-Parallel）问题

**不能混用分片权重与完整 vocab_size！**

当使用 Tensor Parallel (TP) 训练大模型时，`embed_tokens` 和 `output` 权重会在 vocab 维度上分片：

```
完整权重:    [128256, 8192]  ← 单机推理使用
分片权重:    [16032, 8192]   ← TP=8 的每个分片
```

### 检测分片权重

系统会自动检测常见的分片模式：

| 完整 vocab_size | TP Size | 分片 vocab | 示例模型 |
|----------------|---------|-----------|---------|
| 128256 | 8 | 16032 | Llama 3.1 TP=8 |
| 128256 | 4 | 32064 | Llama 3.1 TP=4 |
| 128000 | 8 | 16000 | Llama 2 TP=8 |
| 32000 | 8 | 4000 | Llama 1 TP=8 |

### 解决方案

如果遇到分片权重错误：

**方案 1：合并分片为完整权重（推荐单机推理）**
```python
import torch

# 加载所有分片
shards = [torch.load(f"shard_{i}.pt") for i in range(8)]

# 在 vocab 维度上拼接
embed_full = torch.cat([s["embed_tokens.weight"] for s in shards], dim=0)
output_full = torch.cat([s["output.weight"] for s in shards], dim=0)

# 验证形状
assert embed_full.shape == (128256, 8192)
assert output_full.shape == (128256, 8192)
```

**方案 2：使用支持 TP 的并行引擎**
- Megatron-LM
- vLLM
- DeepSpeed
- 配置 `tensor_parallel_size=8`，每个 rank 加载一个分片

### 错误示例

```
❌ 形状不匹配: embed_tokens.weight
  模块参数形状: (128256, 8192)
  权重文件形状: (16032, 8192)

  ⚠️  检测到 vocab-parallel 分片权重！
    - 权重 vocab 维度: 16032 (分片)
    - 完整 vocab_size: 128256
    - Tensor Parallel (TP) size: 8
    - 当前模型期望: 128256

  ❌ 不能混用分片权重 (16032×...) 与完整 vocab_size (128256)！

  解决方案：
  1. 【推荐】合并分片权重为完整权重
     - 需要收集所有 8 个分片并在 vocab 维度上拼接
     - 合并后 vocab 维度应为: 128256
  2. 使用支持 VocabParallelEmbedding 的并行引擎
     - Megatron-LM / vLLM / DeepSpeed 等
     - 按分片模式加载，每个 rank 加载一个分片
     - 配置 tensor_parallel_size=8
```

### 更新后的关键要点

1. **vocab_size 来源**：从 `params.json` 读取，必须与 checkpoint 一致
2. **embed_tokens**：`num_embeddings` 必须等于 `vocab_size`
3. **output**：`out_features` 必须等于 `vocab_size`
4. **形状对齐**：`embed_tokens.weight.shape[0] == output.weight.shape[0] == vocab_size`
5. **⚠️ 分片检测**：自动检测分片权重（如 TP=8 的 16032）并提供清晰的解决方案
6. **早期检测**：在权重加载时验证形状，避免运行时错误

### 测试验证

```bash
# 测试分片权重检测逻辑
python3 test_sharded_weight_detection.py
```
