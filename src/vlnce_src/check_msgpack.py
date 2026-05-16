
import msgpack_numpy
from PIL import Image
import io
import msgpack
import numpy as np

def encode_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return {
            '__ndarray__': True,
            'dtype': obj.dtype.str,
            'shape': obj.shape,
            'data': obj.tobytes()
        }
    raise TypeError(f"Object of type {type(obj)} is not serializable")

def decode_ndarray(obj):
    if isinstance(obj, dict) and obj.get('__ndarray__'):
        return np.frombuffer(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
    return obj

# packed = msgpack.packb(original, default=encode_ndarray, use_bin_type=True)
# restored = msgpack.unpackb(packed, object_hook=decode_ndarray, raw=False)

def create_mock_trajectory():
    """
    创建一个模拟的轨迹数据，结构与 collect_data 中保存的 transposed_ep 一致：
    [obs_dict, prev_actions_array, oracle_actions_array]
    """
    # 模拟观测字典：包含 RGB 图像（多帧）、指令 token、以及其他信息
    T = 10  # 假设轨迹长度 10 步
    H, W = 224, 224  # 图像尺寸

    # RGB 图像：多帧 uint8 数组，形状 (T, H, W, 3)
    rgb_frames = np.random.randint(0, 255, size=(T, H, W, 3), dtype=np.uint8)

    # 指令 token：假设已经 tokenized，形状 (T, seq_len) 或 (T,)
    instruction_tokens = np.random.randint(0, 1000, size=(T, 20), dtype=np.int64)

    # 其他可能的观测字段
    depth = np.random.rand(T, H, W, 1).astype(np.float32)
    progress = np.random.rand(T, 1).astype(np.float32)

    obs_dict = {
        'rgb': rgb_frames,
        'instruction': instruction_tokens,
        'depth': depth,
        'progress': progress,
        # 可选的 teacher_action（可能在存储时已删除，但这里保留以模拟完整）
        'teacher_action': np.random.randint(0, 10, size=(T, 1), dtype=np.int64),
    }

    # 上一动作和 oracle 动作
    prev_actions = np.random.randint(0, 10, size=(T,), dtype=np.int64)
    oracle_actions = np.random.randint(0, 10, size=(T,), dtype=np.int64)

    return [obs_dict, prev_actions, oracle_actions]

# 只保留 RGB 和动作
def create_simple_mock():
    T, H, W = 10, 224, 224
    rgb = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
    prev_actions = np.random.randint(0, 10, (T,), dtype=np.int64)
    oracle_actions = np.random.randint(0, 10, (T,), dtype=np.int64)
    obs_dict = {'rgb': rgb}
    return [obs_dict, prev_actions, oracle_actions]
    return [obs_dict]
    # return [prev_actions, oracle_actions]


def compare_objects(original, restored):
    """
    递归比较两个对象（支持 dict, list, tuple, numpy.ndarray 等）
    返回是否相等，若不相等打印差异
    """
    if type(original) != type(restored):
        print(f"Type mismatch: {type(original)} vs {type(restored)}")
        return False

    if isinstance(original, np.ndarray):
        if not np.array_equal(original, restored):
            print("Numpy array not equal")
            print("Original shape:", original.shape, "dtype:", original.dtype)
            print("Restored shape:", restored.shape, "dtype:", restored.dtype)
            # 可选：打印少量数据对比
            print("Original first few elements:", original.flat[:5])
            print("Restored first few elements:", restored.flat[:5])
            return False
        return True
    elif isinstance(original, dict):
        if set(original.keys()) != set(restored.keys()):
            print("Dict keys mismatch")
            return False
        for k in original:
            if not compare_objects(original[k], restored[k]):
                print(f"Mismatch at key: {k}")
                return False
        return True
    elif isinstance(original, (list, tuple)):
        if len(original) != len(restored):
            print(f"Length mismatch: {len(original)} vs {len(restored)}")
            return False
        for i, (a, b) in enumerate(zip(original, restored)):
            if not compare_objects(a, b):
                print(f"Mismatch at index: {i}")
                return False
        return True
    else:
        # 标量或其它类型直接比较
        return original == restored

def test_serialization():
    print("创建模拟轨迹数据...")
    original = create_mock_trajectory()
    original = create_simple_mock()

    print("使用 msgpack_numpy.packb 序列化...")
    try:
        # packed = msgpack_numpy.packb(original, use_bin_type=True)
        packed = msgpack.packb(original, default=encode_ndarray, use_bin_type=True)
    except Exception as e:
        print(f"序列化失败: {e}")
        return False

    print(f"序列化后数据大小: {len(packed)} 字节")

    print("使用 msgpack_numpy.unpackb 反序列化...")
    try:
        # restored = msgpack_numpy.unpackb(packed, raw=False)
        restored = msgpack.unpackb(packed, object_hook=decode_ndarray, raw=False)
    except Exception as e:
        print(f"反序列化失败: {e}")
        return False

    print("比较原始数据与恢复后数据...")
    if compare_objects(original, restored):
        print("✅ 测试通过：序列化与反序列化完全一致")
        return True
    else:
        print("❌ 测试失败：数据不一致")
        return False

if __name__ == "__main__":
    test_serialization()
