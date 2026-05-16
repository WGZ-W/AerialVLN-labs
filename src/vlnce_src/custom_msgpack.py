# custom_msgpack.py
import msgpack
import numpy as np

def encode_ndarray(obj):
    """将 numpy 数组转换为可序列化的字典"""
    if isinstance(obj, np.ndarray):
        return {
            '__ndarray__': True,
            'dtype': obj.dtype.str,
            'shape': obj.shape,
            'data': obj.tobytes()
        }
    # 如果对象不是数组，返回原对象（msgpack 会尝试默认序列化）
    return obj

def decode_ndarray(obj):
    """将包含 __ndarray__ 标记的字典恢复为 numpy 数组"""
    if isinstance(obj, dict) and obj.get('__ndarray__'):
        return np.frombuffer(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
    return obj

def packb(data):
    """使用自定义编码序列化数据"""
    return msgpack.packb(data, default=encode_ndarray, use_bin_type=True)

def unpackb(data):
    """使用自定义解码反序列化数据"""
    return msgpack.unpackb(data, object_hook=decode_ndarray, raw=False)