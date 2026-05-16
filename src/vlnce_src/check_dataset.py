import lmdb
import msgpack_numpy

env = lmdb.open('/mnt/sdd/weiguanzhao/AirVLN_ws/DATA/img_features/collect/AirVLN-VLN-3/train', readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        # try:
        data = msgpack_numpy.unpackb(value)
            # 可选：进一步检查 data 的结构是否符合预期
        # except Exception as e:
        #     print(f"Corrupted key: {key}, error: {e}")