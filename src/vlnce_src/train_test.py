import os
import sys
from pathlib import Path
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import gc
import time
import lmdb
import tqdm
import math
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from typing import List, Optional, DefaultDict
import msgpack_numpy

from utils.logger import logger
from utils.utils import get_rank, is_dist_avail_and_initialized, is_main_process, init_distributed_mode
from Model.il_trainer_2 import VLNCETrainer
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import observations_to_image, append_text_to_image, generate_video

from src.common.param import args
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import read_vocab, Tokenizer
import custom_msgpack
import deepspeed


def setup():
    init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False



class DDPIWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
        image_processor=None,
        tokenizer=None,
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.start = 0
        self.end = self.length

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = 0 + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        logger.warning("END init DDP-Dataset \t rank: {} \t start({}) - end({})".format(self.rank, self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        logger.warning("rank: {} \t lmdb load: {} / {}".format(self.rank, i+1, self.preload_size))

                    key_idx = self.load_ordering.pop()
                    key_str = str(self.keys[key_idx]).encode()
                    value = txn.get(key_str)
                    if value is None:
                        continue

                    try:
                        data = custom_msgpack.unpackb(value)
                    except Exception as e:
                        logger.error(f"Corrupted trajectory for key {key_str}, skipping. Error: {e}")
                        continue

                    new_preload.append(data)
                    lengths.append(len(data[0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        rgb_frames = obs['rgb']
        from PIL import Image
        pil_images = [Image.fromarray(frame) for frame in rgb_frames]
        pixel_values = self.image_processor(pil_images, return_tensors='pt')['pixel_values']
        obs['pixel_values'] = pixel_values

        input_ids = torch.from_numpy(obs['instruction']).long()
        obs['input_ids'] = input_ids
        if self.tokenizer.pad_token_id is not None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            obs['attention_mask'] = attention_mask

        for k, v in obs.items():
            if k not in ['pixel_values', 'input_ids', 'attention_mask']:
                obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(self.iter_start, self.iter_end)), self.preload_size)
            )
        )
        return self


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
        image_processor=None,
        tokenizer=None,
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.iter_start = 0
        self.iter_end = self.length
        logger.warning("END init Dataset \t start({}) - end({})".format(self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    key_idx = self.load_ordering.pop()
                    key_str = str(self.keys[key_idx]).encode()
                    value = txn.get(key_str)
                    if value is None:
                        continue

                    try:
                        data = custom_msgpack.unpackb(value)
                    except Exception as e:
                        logger.error(f"Corrupted trajectory for key {key_str}, skipping. Error: {e}")
                        continue

                    if (i+1) % 10 == 0:
                        if self.worker_info is not None:
                            logger.info("{} lmdb load: {} / {}".format(self.worker_info.id, i+1, self.preload_size))
                        else:
                            logger.info("{} lmdb load: {} / {}".format(0, i+1, self.preload_size))

                    new_preload.append(data)
                    lengths.append(len(data[0]))

            if len(new_preload) == 0:
                return self._load_next()

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        rgb_frames = obs['rgb']
        from PIL import Image
        pil_images = [Image.fromarray(frame) for frame in rgb_frames]
        pixel_values = self.image_processor(pil_images, return_tensors='pt')['pixel_values']
        obs['pixel_values'] = pixel_values

        input_ids = torch.from_numpy(obs['instruction']).long()
        obs['input_ids'] = input_ids
        if self.tokenizer.pad_token_id is not None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            obs['attention_mask'] = attention_mask

        for k, v in obs.items():
            if k not in ['pixel_values', 'input_ids', 'attention_mask']:
                obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_info = worker_info
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))
            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = 200
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid][:max_traj_len, ...], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid][:max_traj_len, ...], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


@torch.no_grad()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def initialize_tokenizer():
    if args.tokenizer_use_bert:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        vocab = read_vocab(args.TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    return tok


def initialize_env(split='train'):
    tok = initialize_tokenizer()

    train_env = AirVLNENV(batch_size=args.batchSize, split=split, tokenizer=tok)

    return train_env


def initialize_trainer():
    from gym import spaces
    from airsim_plugin.airsim_settings import AirsimActions

    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
        "instruction": spaces.Discrete(0),
        "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
    })
    action_space = spaces.Discrete(int(len(AirsimActions)))

    from Model.policy_2 import LLaVAPolicy

    vla_path = "/mnt/sdc/weiguanzhao/navila-llama3-8b-8f"

    custom_policy = LLaVAPolicy(vla_path, action_space)

    trainer = VLNCETrainer(
        load_from_ckpt=False,
        observation_space=observation_space,
        action_space=action_space,
        policy=custom_policy,
    )

    logger.info('initialize_trainer over')
    return trainer


def collect_data(data_it=0):
    logger.info(args)

    train_env = initialize_env(split='train')
    trainer = initialize_trainer()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()

    p = 1.0
    beta = 1.0

    with torch.no_grad():
        end_iter = len(train_env.data)
        pbar = None
        pbar_pre_index = 0
        while train_env.index_data < end_iter:
            if pbar_pre_index + train_env.batch_size >= end_iter:
                break

            pbar_pre_index = train_env.index_data
            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            if pbar is None:
                pbar = tqdm.tqdm(total=end_iter)
                pbar.update(train_env.index_data)
            else:
                pbar.update(n=train_env.index_data - pbar_pre_index)

            rnn_states = None  # LLaVAPolicy 不使用 RNN
            prev_actions = torch.zeros(
                train_env.batch_size,
                1,
                dtype=torch.long,
                device=trainer.device,
            )
            not_done_masks = torch.zeros(
                train_env.batch_size,
                1,
                dtype=torch.uint8,
                device=trainer.device,
            )

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            outputs = train_env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)
            batch = preprocess_batch(batch, trainer)

            ended = False

            for t in range(int(args.maxAction) + 1):
                logger.info('{} - {} / {}'.format(int(train_env.index_data)-int(train_env.batch_size), t, end_iter))

                print("Batch keys before act:", batch.keys())
                actions, rnn_states = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                teacher_actions = batch['teacher_action'].long()
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    teacher_actions,
                    actions,
                )

                for i in range(train_env.batch_size):
                    if i in envs_to_pause:
                        continue
                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch['teacher_action'][i].item(),
                        )
                    )

                prev_actions.copy_(actions)

                actions_list = [temp[0] for temp in actions.cpu().numpy()]
                train_env.makeActions(actions_list)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)
                batch = preprocess_batch(batch, trainer)

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

            for i in range(train_env.batch_size):
                if dones[i] and not t >= int(args.maxAction):
                    continue

                ep = episodes[i]
                if len(ep) <= 0:
                    continue

                traj_obs = batch_obs(
                    [step[0] for step in ep],
                    device=torch.device("cpu"),
                )
                del traj_obs['teacher_action']
                for k, v in traj_obs.items():
                    traj_obs[k] = v.numpy()

                transposed_ep = [
                    traj_obs,
                    np.array([step[1] for step in ep], dtype=np.int64),
                    np.array([step[2] for step in ep], dtype=np.int64),
                ]

                train_env.threading_lock_lmdb_features_txn.acquire()
                try:
                    lmdb_key = str(infos[i]['episode_id'])
                    data = custom_msgpack.packb(transposed_ep)
                    train_env.lmdb_features_txn.put(lmdb_key.encode(), data)
                    train_env.lmdb_features_txn.commit()
                    train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                    train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                    train_env.lmdb_collected_keys.add(lmdb_key)
                except Exception as e:
                    logger.error(f"LMDB write failed for key {lmdb_key}: {e}")
                    train_env.lmdb_features_txn.abort()
                    train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                finally:
                    train_env.threading_lock_lmdb_features_txn.release()
                logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                episodes[i] = []
                envs_to_pause.append(i)
                skips[i] = True

    try:
        pbar.close()
    except:
        pass

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass
    logger.info('END data_it: {}'.format(data_it))


def preprocess_batch(batch, trainer):
    image_processor = trainer.policy.image_processor
    tokenizer = trainer.policy.tokenizer

    rgb = batch['rgb']
    rgb_np = rgb.cpu().numpy().astype(np.uint8)
    from PIL import Image
    if rgb_np.ndim == 4:
        pil_images = [Image.fromarray(img).resize((224, 224))
                      for img in rgb_np]
    else:
        pil_images = [Image.fromarray(rgb_np).resize((224, 224))]
    pixel_values = image_processor(pil_images, return_tensors='pt')['pixel_values'].to(trainer.device)
    batch['pixel_values'] = pixel_values

    input_ids = batch['instruction']
    batch['input_ids'] = input_ids
    if tokenizer.pad_token_id is not None:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(trainer.device)
        batch['attention_mask'] = attention_mask

    return batch


def train_vlnce():
    logger.info(args)

    if args.deepspeed:
        deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    if get_rank() == 0:
        writer = SummaryWriter(
            log_dir=str(Path(args.project_prefix) / 'DATA/output/{}/train/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        )
    else:
        writer = None

    trainer = initialize_trainer()

    if args.deepspeed:
        trainable_params = [p for p in trainer.policy.parameters() if p.requires_grad]
        custom_optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        trainer.model_engine, trainer.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=trainer.policy,
            optimizer=custom_optimizer,
            model_parameters=trainable_params,
        )
    else:
        trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=args.lr)

    for dagger_it in range(int(args.dagger_it)):
        step_id = 0
        accumulation_steps = args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1

        lmdb_features_dir = str(Path(args.project_prefix) / 'DATA/img_features/collect/{}/train'.format(args.name))
        assert os.path.exists(str(lmdb_features_dir))

        raw_policy = trainer.policy.module if hasattr(trainer.policy, 'module') else trainer.policy

        # 根据是否分布式选择数据集
        use_distributed_dataset = dist.is_initialized() and dist.get_world_size() > 1
        if use_distributed_dataset:
            dataset = DDPIWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                lmdb_map_size=5.0e12,
                batch_size=args.batchSize,
                image_processor=raw_policy.image_processor,
                tokenizer=raw_policy.tokenizer,
            )
        else:
            dataset = IWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                lmdb_map_size=5.0e12,
                batch_size=args.batchSize,
                image_processor=raw_policy.image_processor,
                tokenizer=raw_policy.tokenizer,
            )

        diter = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batchSize,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
            num_workers=0,
        )

        AuxLosses.activate()
        for epoch in tqdm.trange(int(args.epochs), dynamic_ncols=True):
            batch_cnt = 0
            total_batches = dataset.length // dataset.batch_size if not use_distributed_dataset else (dataset.iter_end - dataset.iter_start) // dataset.batch_size
            for batch in tqdm.tqdm(diter, total=total_batches, leave=False, dynamic_ncols=True):
                (
                    observations_batch,
                    prev_actions_batch,
                    not_done_masks,
                    corrected_actions_batch,
                    weights_batch,
                ) = batch

                observations_batch = {k: v.to(trainer.device, non_blocking=True) for k, v in observations_batch.items()}
                prev_actions_batch = prev_actions_batch.to(trainer.device, non_blocking=True)
                not_done_masks = not_done_masks.to(trainer.device, non_blocking=True)
                corrected_actions_batch = corrected_actions_batch.to(trainer.device, non_blocking=True)
                weights_batch = weights_batch.to(trainer.device, non_blocking=True)

                loss, action_loss, aux_loss = trainer._update_agent(
                    observations_batch, prev_actions_batch, not_done_masks,
                    corrected_actions_batch, weights_batch,
                    step_grad=False,
                    loss_accumulation_scalar=1
                )

                if args.deepspeed:
                    trainer.model_engine.backward(loss)
                    if (batch_cnt + 1) % accumulation_steps == 0:
                        trainer.model_engine.step()
                        trainer.model_engine.zero_grad()
                else:
                    loss.backward()
                    if (batch_cnt + 1) % accumulation_steps == 0:
                        trainer.optimizer.step()
                        trainer.optimizer.zero_grad()

                logger.warning(
                    'dagger_it: {} / {} \t epoch: {} / {} \t batch: {} / {}'.format(
                        dagger_it, args.dagger_it,
                        epoch, args.epochs,
                        batch_cnt, total_batches
                    )
                )

                logger.info(f"train_loss: {loss}")
                logger.info(f"train_action_loss: {action_loss}")
                logger.info(f"train_aux_loss: {aux_loss}")
                logger.info(f"Batches processed: {step_id}.")
                logger.info(f"On DAgger iter {dagger_it}, Epoch {epoch}.")
                logger.info('\n')

                if get_rank() == 0:
                    writer.add_scalar(f"train_loss_iter_{dagger_it}", loss, step_id)
                    writer.add_scalar(f"train_action_loss_iter_{dagger_it}", action_loss, step_id)
                    writer.add_scalar(f"train_aux_loss_iter_{dagger_it}", aux_loss, step_id)

                step_id += 1
                batch_cnt += 1

            if is_main_process():
                save_dir = Path(args.project_prefix) / 'DATA/output/{}/train/checkpoint/{}'.format(args.name, args.make_dir_time)
                save_dir.mkdir(parents=True, exist_ok=True)
                if args.deepspeed:
                    trainer.model_engine.save_checkpoint(save_dir, tag=f"ckpt.{dagger_it}.{epoch}")
                else:
                    if ((dagger_it * args.epochs + epoch)+1) % 5 == 0:
                        trainer.save_checkpoint(f"ckpt.{dagger_it * args.epochs + epoch}.pth", dagger_it, epoch)

            if is_dist_avail_and_initialized():
                dist.barrier()

        if is_main_process():
            trainer.save_checkpoint("ckpt.LAST.pth", dagger_it, epoch)
        AuxLosses.deactivate()


def eval_vlnce():
    logger.info(args)

    writer = TensorboardWriter(
        str(Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        flush_secs=30,
    )

    tok = initialize_tokenizer()

    assert os.path.exists(args.EVAL_CKPT_PATH_DIR), 'The eval file/folder does not exist'
    if os.path.isfile(args.EVAL_CKPT_PATH_DIR):
        from Model.utils.common import get_checkpoint_id

        proposed_index = get_checkpoint_id(args.EVAL_CKPT_PATH_DIR)
        if proposed_index is not None:
            ckpt_idx = proposed_index
        else:
            ckpt_idx = 100000

        _eval_checkpoint(
            checkpoint_path=args.EVAL_CKPT_PATH_DIR,
            writer=writer,
            tok=tok,
            checkpoint_index=ckpt_idx,
        )
        logger.info("END evaluate")
    else:
        from Model.utils.common import poll_checkpoint_folder

        prev_ckpt_ind = -1
        while True:
            current_ckpt = None
            while current_ckpt is None:
                current_ckpt = poll_checkpoint_folder(
                    args.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                )
            logger.info(f"=======current_ckpt: {current_ckpt}=======")
            prev_ckpt_ind += 1

            _eval_checkpoint(
                checkpoint_path=current_ckpt,
                writer=writer,
                tok=tok,
                checkpoint_index=prev_ckpt_ind,
            )

    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)
    logger.info("END evaluate")


def _eval_checkpoint(
    checkpoint_path: str,
    writer,
    tok,
    checkpoint_index: int = 0,
) -> None:
    logger.info(f"checkpoint_path: {checkpoint_path}")

    if args.EVAL_DATASET == 'train':
        train_env = AirVLNENV(batch_size=args.batchSize, split='train', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_unseen', tokenizer=tok)
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNENV(batch_size=args.batchSize, split='test', tokenizer=tok)
    else:
        raise KeyError

    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return

    from Model.policy import LLaVAPolicy

    vla_path = "/mnt/sdc/weiguanzhao/navila-llama3-8b-8f"

    action_space = train_env.action_space
    policy = LLaVAPolicy(vla_path, action_space)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(ckpt["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    trainer = VLNCETrainer(
        load_from_ckpt=False,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        policy=policy,
    )

    trainer.policy.eval()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()
    gc.collect()

    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            rnn_states = None
            prev_actions = torch.zeros(
                train_env.batch_size,
                1,
                dtype=torch.long,
                device=trainer.device,
            )
            not_done_masks = torch.zeros(
                train_env.batch_size,
                1,
                dtype=torch.uint8,
                device=trainer.device,
            )

            rgb_frames = [[] for _ in range(train_env.batch_size)]

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            outputs = train_env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device)
            batch = preprocess_batch(batch, trainer)

            ended = False

            for t in range(int(args.maxAction)):
                logger.info('checkpoint_index:{} \t {} - {} / {} \t {}'.format(checkpoint_index, idx, t, end_iter, not_done_masks.cpu().numpy().reshape((-1,)).tolist()))

                actions, rnn_states = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                    step=t,
                )
                prev_actions.copy_(actions)

                actions = [temp[0] for temp in actions.cpu().numpy()]
                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)
                batch = preprocess_batch(batch, trainer)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

                for i in range(train_env.batch_size):
                    if args.EVAL_GENERATE_VIDEO:
                        frame = observations_to_image(observations[i], infos[i])
                        frame = append_text_to_image(
                            frame, train_env.batch[i]['instruction']['instruction_text']
                        )
                        rgb_frames[i].append(frame)

                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True
                    pbar.update()

                if np.array(dones).all():
                    ended = True
                    break

            for t in range(int(train_env.batch_size)):
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                EVAL_SAVE_EVERY_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results_every/{}'.format(args.name, args.make_dir_time)
                if not os.path.exists(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index))):
                    os.makedirs(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)), exist_ok=True)

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)),
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                if args.EVAL_GENERATE_VIDEO:
                    EVAL_GENERATE_VIDEO_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/videos/{}'.format(args.name, args.make_dir_time)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=rgb_frames[t],
                        episode_id=train_env.batch[t]['episode_id'],
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "ndtw": infos[t]['ndtw'],
                        },
                        tb_writer=writer,
                    )

                logger.info((
                    'result-{} \t' +
                    'distance_to_goal: {} \t' +
                    'success: {} \t' +
                    'ndtw: {} \t' +
                    'sdtw: {} \t' +
                    'path_length: {} \t' +
                    'oracle_success: {} \t' +
                    'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['success'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                ))

    pbar.close()

    EVAL_INTERMEDIATE_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results/{}'.format(args.name, args.make_dir_time)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values())
            / num_episodes
        )

    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = checkpoint_index + 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_{train_env.split}_{k}", v, checkpoint_num)

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


if __name__ == "__main__":
    setup()

    if args.run_type == 'collect':
        collect_data()
    elif args.run_type == 'train':
        train_vlnce()
    elif args.run_type == 'eval':
        eval_vlnce()
    else:
        raise NotImplementedError