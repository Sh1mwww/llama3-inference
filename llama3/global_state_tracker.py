from __future__ import annotations
from typing import List, Optional, Set, Dict, Tuple, Union
import threading
from dataclasses import dataclass
from enum import Enum
import time

class StorageType(Enum):
    HBM = "hbm"
    DRAM = "dram"
    SSD = "ssd"

@dataclass
class BlockInfo:
    batch_idx: int
    layer_idx: int
    block_idx: int
    storage_type: StorageType
    timestamp: float
    importance_score: float = 0.0
    access_count: int = 0

@dataclass
class CopyOpToken:
    kind: str
    bytes: int
    t0: float  # 开始时间

@dataclass
class CopyCounters:
    in_flight: int = 0
    total_bytes: int = 0
    total_ops: int = 0
    last_bw_MBps: float = 0.0
    last_t: float = 0.0

class GlobalStateTracker:
 
    def __init__(self, max_batch: int, layers: int, n_blocks: int):
        self.max_batch = max_batch
        self.layers = layers
        self.n_blocks = n_blocks
        
        self.current_batch = 0
        self.current_layer = 0
        
        self._lock = threading.RLock()
        
        # 存储状态
        # {(batch_idx, layer_idx): set(block_indices)}
        self.hbm_storage: Dict[Tuple[int, int], Set[int]] = {}
        self.dram_storage: Dict[Tuple[int, int], Set[int]] = {}
        self.ssd_storage: Dict[Tuple[int, int], Set[int]] = {}
        
        # Block mapping and metadata
        # {(batch_idx, layer_idx, block_idx): BlockInfo}
        self.block_info: Dict[Tuple[int, int, int], BlockInfo] = {}
        
        # 存储使用统计
        self.storage_stats = {
            StorageType.HBM: {'total_blocks': 0, 'used_blocks': 0, 'capacity_limit': 1000},
            StorageType.DRAM: {'total_blocks': 0, 'used_blocks': 0, 'capacity_limit': 10000},
            StorageType.SSD: {'total_blocks': 0, 'used_blocks': 0, 'capacity_limit': 100000}
        }

        self.copy_stats: Dict[str, CopyCounters] = {
            "weight_h2d": CopyCounters(),
            "kv_h2d":     CopyCounters(),
            "kv_d2h":     CopyCounters(),
        }
        
        # 历史记录
        self.operation_history: List[Dict[str, Union[str, float, int]]] = []
        self.max_history_size = 1000
        
        self.future_batches: List[int] = []
    
    # -----------------------
    # State updates
    # -----------------------
    def set_current_execution(self, batch_idx: int, layer_idx: int):
        """
        设置当前正在执行的batch和layer
        """
        with self._lock:
            self.current_batch = batch_idx
            self.current_layer = layer_idx
            self._log_operation('set_execution', f"batch={batch_idx}, layer={layer_idx}")

    # -----------------------
    # 内部辅助：跨层存在性与定位
    # -----------------------
    def _present_in_any_tier(self, batch_idx: int, layer_idx: int, block_idx: int) -> bool:
        """
        检查给定 block 是否仍存在于任一层级
        """
        key = (batch_idx, layer_idx)
        return (
            (key in self.hbm_storage and block_idx in self.hbm_storage[key]) or
            (key in self.dram_storage and block_idx in self.dram_storage[key]) or
            (key in self.ssd_storage and block_idx in self.ssd_storage[key])
        )

    def _pick_current_location(self, batch_idx: int, layer_idx: int, block_idx: int) -> Optional[StorageType]:
        """
        返回该 block 当前所在层(若允许多层共存,这里可定义优先级: HBM > DRAM > SSD)
        """
        key = (batch_idx, layer_idx)
        if key in self.hbm_storage and block_idx in self.hbm_storage[key]:
            return StorageType.HBM
        if key in self.dram_storage and block_idx in self.dram_storage[key]:
            return StorageType.DRAM
        if key in self.ssd_storage and block_idx in self.ssd_storage[key]:
            return StorageType.SSD
        return None

    def _get_storage_dict(self, storage_type: StorageType) -> Dict[Tuple[int, int], Set[int]]:
        if storage_type == StorageType.HBM:
            return self.hbm_storage
        elif storage_type == StorageType.DRAM:
            return self.dram_storage
        elif storage_type == StorageType.SSD:
            return self.ssd_storage
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def _lower_tier(self, storage_type: StorageType) -> Optional[StorageType]:
        """返回下一级存储层（HBM→DRAM，DRAM→SSD，SSD 无更低层）"""
        if storage_type == StorageType.HBM:
            return StorageType.DRAM
        if storage_type == StorageType.DRAM:
            return StorageType.SSD
        return None

    def _layer_distance(self, layer_idx: int) -> int:
        """
        基于当前执行层 self.current_layer 的“下一次被计算”的距离（0..L-1，环绕）
        - 若 layer_idx == current_layer，则距离为 0（“正在被需要/即将需要”）
        - 距离越大，越晚才会被需要（越优先驱逐）
        """
        if self.layers <= 0:
            return 0
        cur = self.current_layer % self.layers
        l = layer_idx % self.layers
        # 在 0..L-1 的环上计算从 cur 前进到 l 的步数
        return (l - cur) % self.layers

    # -----------------------
    # 存储更新（增/删/清）
    # -----------------------
    def update_storage(self, storage_type: StorageType, batch_idx: int, layer_idx: int, 
                      blocks: List[int], operation: str = 'add', 
                      importance_scores: Optional[List[float]] = None):
        """通用存储更新方法"""
        with self._lock:
            key = (batch_idx, layer_idx)
            storage_dict = self._get_storage_dict(storage_type)
            
            if operation == 'add':
                if key not in storage_dict:
                    storage_dict[key] = set()
                storage_dict[key].update(blocks)
                
                # 更新block详细信息
                now = time.time()
                for i, block_idx in enumerate(blocks):
                    block_key = (batch_idx, layer_idx, block_idx)
                    importance = (
                        importance_scores[i] if (importance_scores and i < len(importance_scores)) else 0.0
                    )
                    
                    if block_key in self.block_info:
                        info = self.block_info[block_key]
                        info.storage_type = storage_type
                        info.importance_score = importance
                        info.access_count += 1
                        info.timestamp = now
                    else:
                        self.block_info[block_key] = BlockInfo(
                            batch_idx=batch_idx,
                            layer_idx=layer_idx,
                            block_idx=block_idx,
                            storage_type=storage_type,
                            timestamp=now,
                            importance_score=importance,
                            access_count=1
                        )
                        
            elif operation == 'remove':
                if key in storage_dict:
                    storage_dict[key].difference_update(blocks)
                    if not storage_dict[key]:
                        del storage_dict[key]
                
                # 安全删除：只有当该块不在任何层时，才从 block_info 删除
                now = time.time()
                for block_idx in blocks:
                    block_key = (batch_idx, layer_idx, block_idx)
                    if block_key in self.block_info:
                        if not self._present_in_any_tier(batch_idx, layer_idx, block_idx):
                            del self.block_info[block_key]
                        else:
                            loc = self._pick_current_location(batch_idx, layer_idx, block_idx)
                            if loc is not None:
                                self.block_info[block_key].storage_type = loc
                            self.block_info[block_key].timestamp = now
                        
            elif operation == 'clear':
                # 清除此层此 (batch, layer) 下的所有块，但仅在不再存在于任何层时才删除 block_info
                if key in storage_dict:
                    to_remove = list(storage_dict[key])
                    now = time.time()
                    for block_idx in to_remove:
                        storage_dict[key].remove(block_idx)
                        block_key = (batch_idx, layer_idx, block_idx)
                        if block_key in self.block_info:
                            if not self._present_in_any_tier(batch_idx, layer_idx, block_idx):
                                del self.block_info[block_key]
                            else:
                                loc = self._pick_current_location(batch_idx, layer_idx, block_idx)
                                if loc is not None:
                                    self.block_info[block_key].storage_type = loc
                                self.block_info[block_key].timestamp = now
                    if not storage_dict[key]:
                        del storage_dict[key]
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            self._update_storage_stats()
            self._log_operation(f'{storage_type.value}_{operation}', 
                              f"batch={batch_idx}, layer={layer_idx}, blocks={blocks}")

    def update_hbm_storage(self, batch_idx: int, layer_idx: int, blocks: List[int], 
                          operation: str = 'add', importance_scores: Optional[List[float]] = None):
        """更新HBM存储状态"""
        self.update_storage(StorageType.HBM, batch_idx, layer_idx, blocks, operation, importance_scores)
    
    def update_dram_storage(self, batch_idx: int, layer_idx: int, blocks: List[int], 
                           operation: str = 'add', importance_scores: Optional[List[float]] = None):
        """更新DRAM存储状态"""
        self.update_storage(StorageType.DRAM, batch_idx, layer_idx, blocks, operation, importance_scores)
    
    def update_ssd_storage(self, batch_idx: int, layer_idx: int, blocks: List[int], 
                          operation: str = 'add', importance_scores: Optional[List[float]] = None):
        """更新SSD存储状态"""
        self.update_storage(StorageType.SSD, batch_idx, layer_idx, blocks, operation, importance_scores)

    # -----------------------
    # 原子迁移
    # -----------------------
    def migrate_blocks(self, batch_idx: int, layer_idx: int, blocks: List[int],
                       src: StorageType, dst: StorageType, importance_scores: Optional[List[float]] = None):
        """原子化从 src 层迁移到 dst 层，并保持 BlockInfo 一致"""
        with self._lock:
            src_dict = self._get_storage_dict(src)
            dst_dict = self._get_storage_dict(dst)
            key = (batch_idx, layer_idx)

            # 从 src 删除
            if key in src_dict:
                src_dict[key].difference_update(blocks)
                if not src_dict[key]:
                    del src_dict[key]

            # 加入 dst
            if key not in dst_dict:
                dst_dict[key] = set()
            dst_dict[key].update(blocks)

            # 统一更新 BlockInfo
            now = time.time()
            for i, block_idx in enumerate(blocks):
                block_key = (batch_idx, layer_idx, block_idx)
                imp = importance_scores[i] if importance_scores and i < len(importance_scores) else None
                if block_key not in self.block_info:
                    self.block_info[block_key] = BlockInfo(
                        batch_idx=batch_idx, layer_idx=layer_idx, block_idx=block_idx,
                        storage_type=dst, timestamp=now, importance_score=(imp or 0.0), access_count=1
                    )
                else:
                    info = self.block_info[block_key]
                    info.storage_type = dst
                    info.timestamp = now
                    if imp is not None:
                        info.importance_score = imp
                    info.access_count += 1

            self._update_storage_stats()
            self._log_operation(
                'migrate',
                f"{src.value}->{dst.value}, batch={batch_idx}, layer={layer_idx}, blocks={blocks}"
            )

    # -----------------------
    # OPT 驱逐：按“距离当前层最远”的 layer 优先踢
    # -----------------------
    def enforce_capacity_opt(
        self,
        storage_type: StorageType,
        target_used: Optional[int] = None,
        batch_filter: Optional[int] = None,
        max_blocks: Optional[int] = None,
    ) -> int:
        """
        当某层使用量超过限制或目标使用量时,按“layer 距离当前层 self.current_layer 最远”的顺序驱逐。
        - storage_type: 要在该层执行驱逐 HBM
        - target_used: 目标使用量；若为空则使用 capacity_limit
        - batch_filter: 若给定，仅驱逐该 batch 的块
        - max_blocks: 限制最多驱逐的 block 数量（可选）
        返回：实际驱逐（迁移）了多少个块
        """
        with self._lock:
            storage_dict = self._get_storage_dict(storage_type)
            lower = self._lower_tier(storage_type)
            if lower is None:
                # 已经是最低层，无法驱逐
                return 0

            # 当前使用与阈值
            used = sum(len(s) for s in storage_dict.values())
            limit = self.storage_stats[storage_type]['capacity_limit']
            target = min(target_used, limit) if (target_used is not None) else limit
            if used <= target:
                return 0

            need_free = used - target
            # 收集候选：[(distance, -importance, timestamp, batch, layer, block)]
            candidates: List[Tuple[int, float, float, int, int, int]] = []
            for (b, l), blocks in storage_dict.items():
                if (batch_filter is not None) and (b != batch_filter):
                    continue
                dist = self._layer_distance(l)
                # dist 越大，越晚用 => 越优先驱逐
                for blk in blocks:
                    info = self.block_info.get((b, l, blk))
                    imp = info.importance_score if info else 0.0
                    ts = info.timestamp if info else 0.0
                    # 注意排序时我们用 (-importance) 使 importance 越小越先出队
                    candidates.append((dist, -imp, ts, b, l, blk))

            if not candidates:
                return 0

            # 排序：distance DESC（远的优先） -> importance ASC（-imp DESC） -> older first（ts ASC）
            candidates.sort(key=lambda x: (x[0], -x[1], -x[2]), reverse=True)

            evicted = 0
            # 分批迁移（按 (batch, layer) 归组，能更高效调用 migrate_blocks）
            idx = 0
            to_move_group: Dict[Tuple[int, int], List[int]] = {}
            while need_free > 0 and idx < len(candidates):
                dist, neg_imp, ts, b, l, blk = candidates[idx]
                # 防止并发变化导致块已不在此层
                if (b, l) in storage_dict and blk in storage_dict[(b, l)]:
                    to_move_group.setdefault((b, l), []).append(blk)
                    evicted += 1
                    need_free -= 1
                    if max_blocks is not None and evicted >= max_blocks:
                        break
                idx += 1

            # 执行迁移
            for (b, l), blks in to_move_group.items():
                self.migrate_blocks(b, l, blks, src=storage_type, dst=lower)

            # 更新统计并记录
            self._update_storage_stats()
            self._log_operation(
                f'{storage_type.value}_enforce_capacity_opt',
                f"moved={evicted} to {lower.value}, target_used={target}, batch_filter={batch_filter}"
            )
            return evicted

    # -----------------------
    # 统计与历史
    # -----------------------
    def _update_storage_stats(self):
        """更新存储统计信息"""
        self.storage_stats[StorageType.HBM]['used_blocks'] = sum(len(blocks) for blocks in self.hbm_storage.values())
        self.storage_stats[StorageType.DRAM]['used_blocks'] = sum(len(blocks) for blocks in self.dram_storage.values())
        self.storage_stats[StorageType.SSD]['used_blocks'] = sum(len(blocks) for blocks in self.ssd_storage.values())
    
    def _log_operation(self, operation: str, details: str):
        """记录操作历史"""
        if len(self.operation_history) >= self.max_history_size:
            self.operation_history.pop(0)
        
        self.operation_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'details': details,
            'current_batch': self.current_batch,
            'current_layer': self.current_layer
        })

    # -----------------------
    # 查询接口
    # -----------------------
    def get_current_state(self):
        """获取当前完整状态"""
        with self._lock:
            return {
                'current_execution': {
                    'batch': self.current_batch,
                    'layer': self.current_layer
                },
                'hbm_data': {k: list(v) for k, v in self.hbm_storage.items()},
                'dram_data': {k: list(v) for k, v in self.dram_storage.items()}, 
                'ssd_data': {k: list(v) for k, v in self.ssd_storage.items()},
                'storage_stats': {k.value: v.copy() for k, v in self.storage_stats.items()},
                'total_blocks_tracked': len(self.block_info)
            }
    
    def get_batch_distribution(self, batch_idx: int):
        """获取特定batch在各存储层级的分布"""
        with self._lock:
            result = {'hbm': {}, 'dram': {}, 'ssd': {}}
            for (b_idx, layer_idx), blocks in self.hbm_storage.items():
                if b_idx == batch_idx:
                    result['hbm'][layer_idx] = sorted(list(blocks))
            for (b_idx, layer_idx), blocks in self.dram_storage.items():
                if b_idx == batch_idx:
                    result['dram'][layer_idx] = sorted(list(blocks))
            for (b_idx, layer_idx), blocks in self.ssd_storage.items():
                if b_idx == batch_idx:
                    result['ssd'][layer_idx] = sorted(list(blocks))
            return result
    
    def get_layer_distribution(self, layer_idx: int):
        """获取特定layer在各存储层级的分布"""
        with self._lock:
            result = {'hbm': {}, 'dram': {}, 'ssd': {}}
            for (batch_idx, l_idx), blocks in self.hbm_storage.items():
                if l_idx == layer_idx:
                    result['hbm'][batch_idx] = sorted(list(blocks))
            for (batch_idx, l_idx), blocks in self.dram_storage.items():
                if l_idx == layer_idx:
                    result['dram'][batch_idx] = sorted(list(blocks))
            for (batch_idx, l_idx), blocks in self.ssd_storage.items():
                if l_idx == layer_idx:
                    result['ssd'][batch_idx] = sorted(list(blocks))
            return result
    
    def get_block_details(self, batch_idx: Optional[int] = None,
                          layer_idx: Optional[int] = None,
                          block_idx: Optional[int] = None) -> List[BlockInfo]:
        """获取block的详细信息"""
        with self._lock:
            results: List[BlockInfo] = []
            for (b_idx, l_idx, blk_idx), info in self.block_info.items():
                if ((batch_idx is None or b_idx == batch_idx) and
                    (layer_idx is None or l_idx == layer_idx) and
                    (block_idx is None or blk_idx == block_idx)):
                    results.append(info)
            return results
    
    def get_storage_utilization(self):
        """获取存储利用率"""
        with self._lock:
            utilization = {}
            for storage_type, stats in self.storage_stats.items():
                if stats['capacity_limit'] > 0:
                    utilization[storage_type.value] = {
                        'used': stats['used_blocks'],
                        'capacity': stats['capacity_limit'],
                        'utilization_rate': stats['used_blocks'] / stats['capacity_limit'],
                        'free': stats['capacity_limit'] - stats['used_blocks']
                    }
            return utilization
    
    def find_blocks_by_importance(self, min_importance: float = 0.0, 
                                 storage_type: Optional[StorageType] = None) -> List[BlockInfo]:
        """根据重要性查找blocks"""
        with self._lock:
            results = []
            for info in self.block_info.values():
                if (info.importance_score >= min_importance and
                    (storage_type is None or info.storage_type == storage_type)):
                    results.append(info)
            return sorted(results, key=lambda x: x.importance_score, reverse=True)
    
    def get_operation_history(self, last_n: int = 10):
        """获取最近的操作历史"""
        with self._lock:
            return self.operation_history[-last_n:] if last_n > 0 else self.operation_history.copy()
    
    # -----------------------
    # 清理
    # -----------------------
    def clear_batch_data(self, batch_idx: int):
        """清除特定batch的所有数据（所有层的 (batch, *)）"""
        with self._lock:
            # 收集要清除的键
            hbm_keys = [k for k in self.hbm_storage if k[0] == batch_idx]
            dram_keys = [k for k in self.dram_storage if k[0] == batch_idx]
            ssd_keys = [k for k in self.ssd_storage if k[0] == batch_idx]

            # 逐层删除并维护 block_info
            for key in hbm_keys:
                blocks = list(self.hbm_storage[key])
                del self.hbm_storage[key]
                for blk in blocks:
                    block_key = (key[0], key[1], blk)
                    if block_key in self.block_info:
                        if not self._present_in_any_tier(key[0], key[1], blk):
                            del self.block_info[block_key]
                        else:
                            loc = self._pick_current_location(key[0], key[1], blk)
                            if loc is not None:
                                self.block_info[block_key].storage_type = loc
                            self.block_info[block_key].timestamp = time.time()

            for key in dram_keys:
                blocks = list(self.dram_storage[key])
                del self.dram_storage[key]
                for blk in blocks:
                    block_key = (key[0], key[1], blk)
                    if block_key in self.block_info:
                        if not self._present_in_any_tier(key[0], key[1], blk):
                            del self.block_info[block_key]
                        else:
                            loc = self._pick_current_location(key[0], key[1], blk)
                            if loc is not None:
                                self.block_info[block_key].storage_type = loc
                            self.block_info[block_key].timestamp = time.time()

            for key in ssd_keys:
                blocks = list(self.ssd_storage[key])
                del self.ssd_storage[key]
                for blk in blocks:
                    block_key = (key[0], key[1], blk)
                    if block_key in self.block_info:
                        if not self._present_in_any_tier(key[0], key[1], blk):
                            del self.block_info[block_key]
                        else:
                            loc = self._pick_current_location(key[0], key[1], blk)
                            if loc is not None:
                                self.block_info[block_key].storage_type = loc
                            self.block_info[block_key].timestamp = time.time()
            
            self._update_storage_stats()
            self._log_operation('clear_batch', f"batch={batch_idx}")

    # -----------------------
    # 打印辅助
    # -----------------------
    def print_current_state(self):
        """打印当前状态的可读格式"""
        state = self.get_current_state()
        print(f"当前执行: Batch {state['current_execution']['batch']}, Layer {state['current_execution']['layer']}")
        
        # 打印存储统计
        print("\n存储统计:")
        for storage_type, stats in state['storage_stats'].items():
            print(f"  {storage_type.upper()}: {stats['used_blocks']} blocks used")
        
        print(f"\n总共跟踪的blocks: {state['total_blocks_tracked']}")
        
        # 打印各存储层级的分布
        print("\nHBM存储:")
        for (batch, layer), blocks in state['hbm_data'].items():
            print(f"  Batch {batch}, Layer {layer}: {blocks}")
        
        print("\nDRAM存储:")
        for (batch, layer), blocks in state['dram_data'].items():
            print(f"  Batch {batch}, Layer {layer}: {blocks}")
            
        print("\nSSD存储:")
        for (batch, layer), blocks in state['ssd_data'].items():
            print(f"  Batch {batch}, Layer {layer}: {blocks}")
    
    def print_storage_utilization(self):
        """打印存储利用率"""
        utilization = self.get_storage_utilization()
        print("\n存储利用率:")
        for storage_type, info in utilization.items():
            print(f"  {storage_type.upper()}: {info['used']}/{info['capacity']} "
                  f"({info['utilization_rate']:.1%}) - 剩余: {info['free']}")

    # -----------------------
    # Future batches
    # -----------------------
    def register_future_batch(self, batch_order: List[int]):
        """
        一次性注册接下来要跑的 batch 顺序
        """
        with self._lock:
            self.future_batches = batch_order.copy()
            self._log_operation('register_future_batches', f"batches={batch_order}")
    
    def get_future_batches(self, offset: int = 1) -> Optional[List[int]]:
        """
        向前看第 offset 个 batch; offset=1 ==> "下一个 batch"
        返回列表（从该位置开始的后续顺序），若不存在返回 None
        """
        with self._lock:
            idx = offset - 1
            return self.future_batches[idx:] if 0 <= idx < len(self.future_batches) else None
    
    def get_next_batch(self, offset: int = 1) -> Optional[int]:
        """返回 offset 之后**单个** batch 索引；默认就是'下一个 batch'"""
        with self._lock:
            if not self.future_batches:
                return None
            try:
                current_pos = self.future_batches.index(self.current_batch)
                next_pos = current_pos + offset
                return self.future_batches[next_pos] if next_pos < len(self.future_batches) else None
            except (ValueError, IndexError):
                return None

    # -----------------------
    # Copy 统计（使用单调时钟更稳）
    # -----------------------
    def start_copy(self, kind: str, bytes_cnt: int) -> CopyOpToken:
        cs = self.copy_stats.get(kind)
        if cs is None:
            self.copy_stats[kind] = cs = CopyCounters()
        cs.in_flight += 1
        return CopyOpToken(kind=kind, bytes=bytes_cnt, t0=time.perf_counter())

    def end_copy(self, token: CopyOpToken):
        cs = self.copy_stats.get(token.kind)
        if cs is None:
            return
        cs.in_flight = max(0, cs.in_flight - 1)
        dt = max(1e-6, time.perf_counter() - token.t0)
        cs.total_ops += 1
        cs.total_bytes += token.bytes
        cs.last_bw_MBps = (token.bytes / 1e6) / dt
        cs.last_t = time.time()

    def inflight(self, kind: str) -> int:
        cs = self.copy_stats.get(kind)
        return cs.in_flight if cs else 0

    def last_bw(self, kind: str) -> float:
        cs = self.copy_stats.get(kind)
        return cs.last_bw_MBps if cs else 0.0

    def prefer_launch_kv_d2h_now(self) -> bool:
        """简单决策信号：当权重H2D不在跑时，更适合发起KV的D2H"""
        return self.inflight("weight_h2d") == 0


# 全局实例
_global_tracker: Optional[GlobalStateTracker] = None

def get_global_tracker() -> Optional[GlobalStateTracker]:
    """获取全局跟踪器实例"""
    global _global_tracker
    return _global_tracker

def init_global_tracker(max_batch: int, layers: int, n_blocks: int) -> GlobalStateTracker:
    """初始化全局跟踪器"""
    global _global_tracker
    _global_tracker = GlobalStateTracker(max_batch, layers, n_blocks)
    return _global_tracker

def reset_global_tracker():
    """重置全局跟踪器"""
    global _global_tracker
    _global_tracker = None
    
def get_current_batch() -> Optional[int]:
    """获取当前执行的batch索引"""
    tracker = get_global_tracker()
    return tracker.current_batch if tracker else None

def get_future_batches(offset: int = 1) -> Optional[List[int]]:
    """获取未来的batch列表"""
    tracker = get_global_tracker()
    return tracker.get_future_batches(offset) if tracker else None
