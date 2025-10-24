# 设备一致性检查和修复

## 问题描述

**nn.Embedding/F.embedding 的设备要求：**

> input tokens 与 embed.weight **必须在同一设备**（CPU 或同一块 GPU）

如果设备不一致，会出现典型的设备不匹配错误：
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

---

## 验证方法

### 快速检查（按照 PyTorch 官方要求）

```python
print(len(tokenizer))                             # 128256
print(model.embed_tokens.num_embeddings)          # 128256
print(model.embed_tokens.weight.shape)            # torch.Size([128256, 8192])
print(model.embed_tokens.weight.device)           # cuda:0 (或 cpu，但必须与 input 一致)
```

### 使用内置验证函数

```python
# 方法 1: 使用 Transformer.print_device_info()
model.print_device_info(tokenizer)

# 方法 2: 使用独立验证脚本
from verify_model_device import verify_model_device
verify_model_device(model, tokenizer)
```

---

## 代码修复

### 1. Transformer.forward() 自动设备转换 (llama3/model.py:65-85)

```python
def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
    # ★ 使用 embed_tokens 的实际设备作为目标设备
    # 这是设备一致性的关键：所有计算必须在 embed_tokens.weight 所在的设备上进行
    tgt_dev = self.embed_tokens.weight.device

    # ★ 设备一致性检查：input tokens 必须与 embed.weight 在同一设备
    # nn.Embedding/F.embedding 要求：input 和 weight 必须在同一设备（CPU 或同一块 GPU）
    if tokens.device != tgt_dev:
        # 自动转换到目标设备
        if tokens.device.type == "cpu" and not tokens.is_pinned():
            tokens = tokens.pin_memory()  # 加速 H2D 传输
        tokens = tokens.to(tgt_dev, non_blocking=True)

    # 确保 dtype 正确
    if tokens.dtype != torch.long:
        tokens = tokens.long()

    # 执行 embedding 查表：此时 tokens 和 weight 已在同一设备
    h = self.embed_tokens(tokens)

    # 验证 embedding 输出设备（调试用）
    assert h.device == dev, f"Embedding 输出设备不一致: {h.device} != {dev}"
```

### 2. 添加 print_device_info() 方法 (llama3/model.py:19-75)

```python
def print_device_info(self, tokenizer=None):
    """
    打印设备和形状信息，用于调试设备一致性问题
    """
    print("=" * 80)
    print("Transformer 设备一致性检查")
    print("=" * 80)

    if tokenizer is not None:
        print(f"Tokenizer vocab_size:           {len(tokenizer)}")

    print(f"embed_tokens.num_embeddings:    {self.embed_tokens.num_embeddings}")
    print(f"embed_tokens.weight.shape:      {self.embed_tokens.weight.shape}")
    print(f"embed_tokens.weight.device:     {self.embed_tokens.weight.device}")

    print(f"output.out_features:            {self.output.out_features}")
    print(f"output.weight.device:           {self.output.weight.device}")

    # 验证一致性
    # ...
```

---

## 测试验证

### 测试 1: 基础设备一致性

```bash
python3 test_device_consistency_lite.py
```

输出：
```
✅ 测试 1: CPU 上的 embedding（设备一致）
   embed.weight.device: cpu
   tokens.device:       cpu
   ✅ 成功

✅ 测试 2: CUDA 上的 embedding（设备一致）
   embed.weight.device: cuda:0
   tokens.device:       cuda:0
   ✅ 成功

✅ 测试 3: 设备不匹配（应该失败）
   embed.weight.device: cuda:0
   tokens.device:       cpu
   ✅ 预期失败: RuntimeError

✅ 测试 4: 自动设备转换（正确做法）
   tokens: cpu -> cuda:0
   ✅ 成功
```

### 测试 2: 模型验证

```bash
python3 verify_model_device.py
```

输出：
```
len(tokenizer):                     128256
model.embed_tokens.num_embeddings:  128256
model.embed_tokens.weight.shape:    torch.Size([128256, 8192])
model.embed_tokens.weight.device:   cuda:0

✅ 所有检查通过！
```

---

## 设备一致性规则

### ✅ 允许的组合

| embed.weight 设备 | input tokens 设备 | 结果 |
|------------------|------------------|------|
| CPU | CPU | ✅ 正常 |
| cuda:0 | cuda:0 | ✅ 正常 |
| cuda:1 | cuda:1 | ✅ 正常 |

### ❌ 禁止的组合

| embed.weight 设备 | input tokens 设备 | 结果 |
|------------------|------------------|------|
| cuda:0 | CPU | ❌ RuntimeError |
| CPU | cuda:0 | ❌ RuntimeError |
| cuda:0 | cuda:1 | ❌ RuntimeError |

### ✅ 解决方案：自动转换

```python
# 方案 1: 在 forward 中自动转换（推荐）
tgt_dev = model.embed_tokens.weight.device
tokens = tokens.to(tgt_dev)
embeddings = model.embed_tokens(tokens)

# 方案 2: 调用方提前转换
tokens = tokens.to(model.embed_tokens.weight.device)
output = model(tokens, start_pos)
```

---

## 数据流验证

```
用户输入 tokens (CPU):
  ↓
自动检测 embed.weight.device (cuda:0)
  ↓
自动转换: tokens.to(cuda:0)
  ↓
embed_tokens(tokens):
  - tokens.device:        cuda:0  ✅
  - embed.weight.device:  cuda:0  ✅
  - 设备一致 → 成功执行
  ↓
embeddings (cuda:0)
  ↓
transformer layers (cuda:0)
  ↓
output projection (cuda:0)
  ↓
logits (cuda:0)
```

---

## 关键要点

1. ✅ **nn.Embedding 要求**：input 和 weight 必须在同一设备
2. ✅ **Transformer.forward()**：自动将 tokens 转换到 embed.weight 所在设备
3. ✅ **验证工具**：
   - `model.print_device_info(tokenizer)` - 内置方法
   - `verify_model_device(model, tokenizer)` - 独立脚本
4. ✅ **调试打印**（按 PyTorch 官方要求）：
   ```python
   print(len(tokenizer))
   print(model.embed_tokens.num_embeddings)
   print(model.embed_tokens.weight.shape)
   print(model.embed_tokens.weight.device)
   ```
5. ✅ **早期检测**：在模型初始化后立即验证设备一致性

---

## 参考文档

- [PyTorch nn.Embedding 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch F.embedding 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html)

> **注意**：nn.Embedding 和 F.embedding 都要求 input 和 weight 在同一设备，这是 PyTorch 的基本约束。

---

## 相关文件

- [llama3/model.py](llama3/model.py) - 模型定义，包含自动设备转换逻辑
- [test_device_consistency_lite.py](test_device_consistency_lite.py) - 轻量级设备测试
- [verify_model_device.py](verify_model_device.py) - 设备验证工具
