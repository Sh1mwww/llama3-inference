from __future__ import annotations
from typing import List, Optional, Set, Dict, Tuple, Union
import numpy as np
import threading
from dataclasses import dataclass
from enum import Enum

class StorageType(Enum):
    """存储类型枚举"""
    HBM = "hbm"
    DRAM = "dram"
    SSD = "ssd"

@dataclass
class BlockInfo:
    """Block信息数据类"""
    batch_idx: int
    layer_idx: int
    block_idx: int
    storage_type: StorageType
    timestamp: float
    importance_score: float = 0.0
    access_count: int = 0

class GlobalStateTracker:
    """全局状态跟踪器，记录当前执行状态和各存储层级的数据分布"""
    
    def __init__(self, max_batch: int, layers: int, n_blocks: int):
        self.max_batch = max_batch
        self.layers = layers
        self.n_blocks = n_blocks
        
        # 当前执行状态
        self.current_batch = 0
        self.current_layer = 0
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 各存储层级的状态映射
        # 格式: {(batch_idx, layer_idx): set(block_indices)}
        self.hbm_storage = {}  # GPU HBM中的数据
        self.dram_storage = {}  # CPU DRAM中的数据  
        self.ssd_storage = {}   # SSD中的数据
        
        # Block详细信息映射
        # 格式: {(batch_idx, layer_idx, block_idx): BlockInfo}
        self.block_info = {}
        
        # 存储使用统计
        self.storage_stats = {
            StorageType.HBM: {'total_blocks': 0, 'used_blocks': 0, 'capacity_limit': 1000},
            StorageType.DRAM: {'total_blocks': 0, 'used_blocks': 0, 'capacity_limit': 10000},
            StorageType.SSD: {'total_blocks': 0, 'used_blocks': 0, 'capacity_limit': 100000}
        }
        
        # 历史记录
        self.operation_history = []
        self.max_history_size = 1000
        
        # 未来批次列表
        self.future_batches: List[int] = []
        
        # ZigZag调度相关
        self.zigzag_enabled = False
        self.zigzag_schedule: List[Tuple[int, int]] = []  # [(batch_idx, layer_idx), ...]
        self.zigzag_current_index = 0  # 当前执行的调度索引
        self.zigzag_completed_executions: List[Tuple[int, int]] = []  # 已完成的执行 
    
    def set_current_execution(self, batch_idx: int, layer_idx: int):
        """设置当前正在执行的batch和layer"""
        with self._lock:
            self.current_batch = batch_idx
            self.current_layer = layer_idx
            self._log_operation('set_execution', f"batch={batch_idx}, layer={layer_idx}")
    
    def update_storage(self, storage_type: StorageType, batch_idx: int, layer_idx: int, 
                      blocks: List[int], operation: str = 'add', 
                      importance_scores: List[float] = None):
        """通用存储更新方法"""
        with self._lock:
            key = (batch_idx, layer_idx)
            
            if storage_type == StorageType.HBM:
                storage_dict = self.hbm_storage
            elif storage_type == StorageType.DRAM:
                storage_dict = self.dram_storage
            elif storage_type == StorageType.SSD:
                storage_dict = self.ssd_storage
            else:
                raise ValueError(f"Unknown storage type: {storage_type}")
            
            if operation == 'add':
                if key not in storage_dict:
                    storage_dict[key] = set()
                storage_dict[key].update(blocks)
                
                # 更新block详细信息
                for i, block_idx in enumerate(blocks):
                    block_key = (batch_idx, layer_idx, block_idx)
                    importance = importance_scores[i] if importance_scores and i < len(importance_scores) else 0.0
                    
                    if block_key in self.block_info:
                        # 更新现有信息
                        self.block_info[block_key].storage_type = storage_type
                        self.block_info[block_key].importance_score = importance
                        self.block_info[block_key].access_count += 1
                    else:
                        # 创建新的block信息
                        import time
                        self.block_info[block_key] = BlockInfo(
                            batch_idx=batch_idx,
                            layer_idx=layer_idx,
                            block_idx=block_idx,
                            storage_type=storage_type,
                            timestamp=time.time(),
                            importance_score=importance,
                            access_count=1
                        )
                        
            elif operation == 'remove':
                if key in storage_dict:
                    storage_dict[key] -= set(blocks)
                    if not storage_dict[key]:
                        del storage_dict[key]
                
                # 从block信息中移除
                for block_idx in blocks:
                    block_key = (batch_idx, layer_idx, block_idx)
                    if block_key in self.block_info:
                        del self.block_info[block_key]
                        
            elif operation == 'clear':
                if key in storage_dict:
                    # 清除block信息
                    for block_idx in storage_dict[key]:
                        block_key = (batch_idx, layer_idx, block_idx)
                        if block_key in self.block_info:
                            del self.block_info[block_key]
                    del storage_dict[key]
            
            self._update_storage_stats()
            self._log_operation(f'{storage_type.value}_{operation}', 
                              f"batch={batch_idx}, layer={layer_idx}, blocks={blocks}")
    
    def update_hbm_storage(self, batch_idx: int, layer_idx: int, blocks: List[int], 
                          operation: str = 'add', importance_scores: List[float] = None):
        """更新HBM存储状态"""
        self.update_storage(StorageType.HBM, batch_idx, layer_idx, blocks, operation, importance_scores)
    
    def update_dram_storage(self, batch_idx: int, layer_idx: int, blocks: List[int], 
                           operation: str = 'add', importance_scores: List[float] = None):
        """更新DRAM存储状态"""
        self.update_storage(StorageType.DRAM, batch_idx, layer_idx, blocks, operation, importance_scores)
    
    def update_ssd_storage(self, batch_idx: int, layer_idx: int, blocks: List[int], 
                          operation: str = 'add', importance_scores: List[float] = None):
        """更新SSD存储状态"""
        self.update_storage(StorageType.SSD, batch_idx, layer_idx, blocks, operation, importance_scores)
    
    def _update_storage_stats(self):
        """更新存储统计信息"""
        self.storage_stats[StorageType.HBM]['used_blocks'] = sum(len(blocks) for blocks in self.hbm_storage.values())
        self.storage_stats[StorageType.DRAM]['used_blocks'] = sum(len(blocks) for blocks in self.dram_storage.values())
        self.storage_stats[StorageType.SSD]['used_blocks'] = sum(len(blocks) for blocks in self.ssd_storage.values())
    
    def _log_operation(self, operation: str, details: str):
        """记录操作历史"""
        import time
        if len(self.operation_history) >= self.max_history_size:
            self.operation_history.pop(0)
        
        self.operation_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'details': details,
            'current_batch': self.current_batch,
            'current_layer': self.current_layer
        })
    
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
    
    def get_block_details(self, batch_idx: int = None, layer_idx: int = None, block_idx: int = None):
        """获取block的详细信息"""
        with self._lock:
            results = []
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
                                 storage_type: StorageType = None):
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
    
    def clear_batch_data(self, batch_idx: int):
        """清除特定batch的所有数据"""
        with self._lock:
            keys_to_remove = []
            
            # 清除各存储层级的数据
            for key in self.hbm_storage:
                if key[0] == batch_idx:
                    keys_to_remove.append(('hbm', key))
            for key in self.dram_storage:
                if key[0] == batch_idx:
                    keys_to_remove.append(('dram', key))
            for key in self.ssd_storage:
                if key[0] == batch_idx:
                    keys_to_remove.append(('ssd', key))
            
            for storage_type, key in keys_to_remove:
                if storage_type == 'hbm':
                    del self.hbm_storage[key]
                elif storage_type == 'dram':
                    del self.dram_storage[key]
                elif storage_type == 'ssd':
                    del self.ssd_storage[key]
            
            # 清除block详细信息
            block_keys_to_remove = [k for k in self.block_info.keys() if k[0] == batch_idx]
            for k in block_keys_to_remove:
                del self.block_info[k]
            
            self._update_storage_stats()
            self._log_operation('clear_batch', f"batch={batch_idx}")
    
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

    def register_future_batch(self, batch_idx: List[int]):
        """
        一次性注册接下来要跑的 batch 顺序
        """
        with self._lock:
           self.future_batches = batch_idx.copy()
           self._log_operation('register_future_batches', f"batches={batch_idx}")
    
    def get_future_batches(self, offset: int = 1) -> List[int]:
        """
        向前看第 offset 个 batch; offset=1 ==> "下一个 batch"
        """
        with self._lock:
            idx = offset - 1
            return self.future_batches[idx:] if 0 <= idx < len(self.future_batches) else None
    
    def get_next_batch(self, offset: int = 1) -> Optional[int]:
        """返回 offset 之后**单个** batch 索引；默认就是"下一个 batch"""
        with self._lock:
            if not self.future_batches:
                return None
            
            # 找到当前batch在future_batches中的位置
            try:
                current_pos = self.future_batches.index(self.current_batch)
                next_pos = current_pos + offset
                return self.future_batches[next_pos] if next_pos < len(self.future_batches) else None
            except (ValueError, IndexError):
                # 如果当前batch不在future_batches中，或索引越界
                return None
    
# 全局实例
_global_tracker = None

def get_global_tracker():
    """获取全局跟踪器实例"""
    global _global_tracker
    return _global_tracker

def init_global_tracker(max_batch: int, layers: int, n_blocks: int):
    """初始化全局跟踪器"""
    global _global_tracker
    _global_tracker = GlobalStateTracker(max_batch, layers, n_blocks)
    return _global_tracker

def reset_global_tracker():
    """重置全局跟踪器"""
    global _global_tracker
    _global_tracker = None
    
def get_current_batch():
    """获取当前执行的batch索引"""
    tracker = get_global_tracker()
    return tracker.current_batch if tracker else None


def get_future_batches(offset: int = 1) -> List[int]:
    """获取未来的batch列表"""
    tracker = get_global_tracker()
    return tracker.get_future_batches(offset) if tracker else None