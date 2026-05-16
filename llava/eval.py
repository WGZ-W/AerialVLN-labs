
from unrealcv import Client
from typing import Union, List, Dict, Any
import cv2
import io
import time
import math
import subprocess, threading
import airsim
from common import *
import psutil
import requests
import random
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import os, json
# from extern.hf.configuration_prismatic import OpenFlyConfig
# from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
# from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# from model.load_model import OpenFly
# from model.vision_backbone import DinoSigLIPViTBackbone
# from model.llm_backbone import LLaMa2LLMBackbone
from llava.model import *
from llava.model.action_tokenizer import ActionTokenizer





os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# AutoConfig.register("openvla", OpenFlyConfig)
# AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
# AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
# AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)



def kill_env_process(keyword):
    result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
    cr_pid = result.stdout.decode().strip()
    if len(cr_pid) > 0:
        subprocess.run(['kill', '-9', cr_pid])

class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(10)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def _init_airsim_sim(self):
        env_dir = "../envs/airsim/" + self.env_name
        # env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
                                  airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)))
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.1)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type = 'color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
        response = responses[0]
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='color'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)
        print("Image saved")


class UEBridge:
    def __init__(self, ue_ip, ue_port, env_name):
        self.kill_failed_process()
        time.sleep(10)

        # port = self.find_available_port()

        port = random.randint(9000, 9100)
        print(f"Available port: {port}")
        self.modify_port_in_ini(port, env_name)
        ue_port = port

        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_ue_sim)
        self._sim_thread.start()
        time.sleep(15)

        self._client = Client((ue_ip, ue_port))
        self._connection_check()

        self._camera_init()

        # self._drone_init()  
        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def find_available_port(self):
        port = 9000
        while True:
            result = subprocess.run(['lsof', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            netstat_output = result.stdout.decode()

            if f'PID' not in netstat_output:
                return port
            port += 1

    def modify_port_in_ini(self, port, ue_env_name):
        # ini_file = f"envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"
        ini_file = f"../envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"

        with open(ini_file, 'r') as file:
            lines = file.readlines()

        with open(ini_file, 'w') as file:
            for line in lines:
                if line.startswith("Port="):
                    file.write(f"Port={port}\n")
                else:
                    file.write(line)

    def kill_failed_process(self):
        result = subprocess.run(['pgrep', '-n', 'CrashReport'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

        result = subprocess.run(['pgrep', '-n', 'CitySample'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

    def _init_ue_sim(self):
        env_dir = "../envs/ue/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = ["bash", f"{env_dir}/CitySample.sh"]

        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)
        time.sleep(2)

    def __del__(self):
        self._client.disconnect()

    def _connection_check(self):
        '''Check if connected'''
        if self._client.connect():
            print('UnrealCV connected successfully')
        else:
            print('UnrealCV is not connected')
            exit()

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        '''Set camera position'''
        x = x * 100
        y = - y * 100
        z = z * 100
        camera_settings = {
            'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'pitch': pitch, 'yaw': -yaw, 'roll': roll}
        }

        self._client.request('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        self._client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        print('camera_settings', camera_settings)

    def _camera_init(self):
        '''Camera initialization'''
        time.sleep(2)
        self._client.request('vset /cameras/spawn')
        self._client.request('vset /camera/1/size 1920 1080')
        time.sleep(2)
        self.set_camera_pose(150, 400, 15, 0, 0, 0)  # Initial position
        time.sleep(2)

    def get_camera_data(self, camera_type = 'lit'):
        valid_types = {'lit', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'lit':
            data = self._client.request('vget /camera/1/lit png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'object_mask':
            data = self._client.request('vget /camera/1/object_mask png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'depth':
            data = self._client.request('vget /camera/1/depth npy')
            depth_np = np.load(io.BytesIO(data))
            return depth_np  # Return depth data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='lit'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)

class GSBridge:  
    def __init__(self, env_name):  
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_gs_sim)
        self._sim_thread.start()
        self.url = "http://localhost:18080/render"
        time.sleep(10)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def _init_gs_sim(self):
        # dataset_dir = "envs/gs/" + self.env_name  
        dataset_dir = "/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/nwpu02"
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"  
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Specified directory {dataset_dir} does not exist")
        command = [
            gs_vis_tool_dir + "install/bin/SIBR_gaussianHierarchyViewer_app",
            "--path", f"{dataset_dir}/camera_calibration/aligned",
            "--scaffold", f"{dataset_dir}/output/scaffold/point_cloud/iteration_30000",
            "--model-path", f"{dataset_dir}/output/merged.hier",
            "--images-path", f"{dataset_dir}/camera_calibration/rectified/images"
        ]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def transform_euler_to_new_frame(self, roll, pitch, yaw):
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        new_R = np.dot(transformation_matrix, R)
        new_roll, new_pitch, new_yaw = rotation_matrix_to_euler_angles(new_R)
        return new_roll, new_pitch, new_yaw
    
    def rotation_matrix_roll(self, roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def rotation_matrix_pitch(self, pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def rotation_matrix_yaw(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def transform_to_camera_frame(self, roll, pitch, yaw):
        R_roll = self.rotation_matrix_roll(roll)
        R_pitch = self.rotation_matrix_pitch(pitch)
        R_yaw = self.rotation_matrix_yaw(yaw)
        R_combined = np.dot(R_pitch, np.dot(R_yaw, R_roll))
        QW, QX, QY, QZ = rotation_matrix_to_quaternion(R_combined)
        print(f"QW: {QW}, QX: {QX}, QY: {QY}, QZ: {QZ}")
        transformation_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        new_R = np.dot(transformation_matrix, R_combined)
        QW_new, QX_new, QY_new, QZ_new = rotation_matrix_to_quaternion(new_R)
        return QW_new, QX_new, QY_new, QZ_new

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
        data = {
            "camera": camera_params,
            "image": image_data,
            "path": path_params
        }
        print(data)
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Request successful!")
                print(response.text) 
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(20)

    def process_camera_data(self, file_path):
        pass



def get_images(lst,if_his,step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]

def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result

def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):

    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)
        
    prompt = text
    inputs = processor(prompt, images).to("cuda:1", dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action


def get_action2(policy, image_list, text, his, if_his=False, his_step=0):
    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)

    prompt = text
    # inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
    action = policy.predict_action(images, prompt, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action


def get_action3(model, image_list, text, norm_stats, action_tokenizer, same_seq=False):
    # Otherwise, generate new actions using the policy

    image = image_list[-1]

    if isinstance(image, Image.Image):
        pass
    else:
        image = Image.fromarray(image)

    action = model.predict_action(image, text,
                                  norm_stats=norm_stats,
                                  action_tokenizer=action_tokenizer,
                                  unnorm_key="vlnv1",
                                  same_seq=same_seq
                                  )
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]


def flatten_state_dict(model_state_dicts):
    """
    把 {"vision_backbone": {"layer.weight": ...}, "llm_backbone": {...}}
    转换为 {"vision_backbone.layer.weight": ..., "llm_backbone.xxx": ...}
    """
    flat = {}
    for mkey, subdict in model_state_dicts.items():
        for subkey, tensor in subdict.items():
            flat[f"{mkey}.{subkey}"] = tensor
    return flat


def main():

    # eval_info = "../configs/eval_test.json"
    # eval_info = "../configs/eval_test2.json"
    # eval_info = "../configs/eval_test_airsim_4.json"
    # eval_info = "../configs/eval.json"
    eval_info = "../configs/airsim16_entries.json" # 3.96
    # eval_info = "../configs/eval_test4.json"

    f = open(eval_info, 'r')
    all_eval_info = json.loads(f.read())
    f.close()
    
    # Load model
    # model_name_or_path="/mnt/sdc/weiguanzhao/openfly-agent-7b"
    # processor = AutoProcessor.from_pretrained(model_name_or_path)
    # policy = AutoModelForVision2Seq.from_pretrained(
    #     model_name_or_path,
    #     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # ).to("cuda:0")

    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila-llama3-8b-8f+b2+lr-0.0005+lora-r32+dropout-0.0"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b2+lr-0.0005+lora-r32+dropout-0.0"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln6/navila+b2+lr-0.0005+lora-r32+dropout-0.0"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"

    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"

    # Lora baseline ??? Baseline
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln-buffer-100000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln-buffer-100000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln-buffer-100000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+15000" # Now

    # This one is my result Mamba
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-buffer-10000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000" # Now
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-buffer-10000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000" # Now
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000" # Now

    # Key
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key3/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7500" # Now
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key3/navila+b1+lr-0.0001+lora-r32+dropout-0.05+2500"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+15000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key2/navila+b1+lr-1e-05+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+20000"

    # Key + Mamba
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key-mamba/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7500"


    # KS
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+2000"


    # Random KS
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+2500"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+5000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7500"
    model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+10000"




    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"



    config = LlavaLlamaConfig.from_pretrained(model_path, resume=False)
    if getattr(config, "resume_path", None) is not None:
        config.resume_path = model_path

    # model = LlavaLlamaModel(
    #     config=config,
    #     attn_implementation="flash_attention_2",
    #     model_max_length=4096,
    #     # model_max_length=2048,
    # ).to("cuda:1")

    model = LlavaLlamaModel2(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=4096,
        # model_max_length=2048,
    ).to("cuda:1")

    model.eval()

    # processor = MultiModalProcessor(model)
    # norm_stats = None
    action_tokenizer = ActionTokenizer(model.tokenizer)


    # 自己训练的 pt 模型导入
    dataset_statistics_path = "/mnt/sdc/weiguanzhao/dataset_statistics.json"
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        # policy.norm_stats = norm_stats
        model.norm_stats = norm_stats

    # vision_backbone = DinoSigLIPViTBackbone(
    #     image_resize_strategy="resize-naive",
    #     default_image_size=224,
    #     grid_size=16,
    # )
    # image_transform = vision_backbone.image_transform
    #
    # llm_backbone = LLaMa2LLMBackbone(
    #     llm_max_length=2048,
    #     hf_token="",
    #     inference_mode=True,
    # )
    # tokenizer = llm_backbone.get_tokenizer()
    # action_tokenizer = ActionTokenizer(tokenizer)
    #
    # policy = OpenFly(
    #     model_id="openfly",
    #     vision_backbone=vision_backbone,
    #     llm_backbone=llm_backbone,
    #     norm_stats=norm_stats,
    #     action_tokenizer=action_tokenizer,
    #     arch_specifier="gelu-mlp",
    # ).to("cuda:0")
    #
    # state_dict = torch.load("/mnt/sda/wgz/openfly2/checkpoints/step-095000-epoch-03-loss=0.1596.pt", map_location="cpu")
    # state_dict = torch.load("/mnt/sda/wgz/runs/n0+b12+x7/checkpoints/step-010000-epoch-03-loss=0.0884.pt", map_location="cpu")
    # state_dict = torch.load("/mnt/sda/wgz/runs/n0+b12+x7/checkpoints/step-015000-epoch-05-loss=0.0211.pt", map_location="cpu")
    # nested_state = state_dict["model"]  # 这里是 vision_backbone/llm_backbone/projector 分组
    # flat_state = flatten_state_dict(nested_state)
    # policy.load_state_dict(flat_state)
    # policy.eval()



    # Test metrics
    acc = 0
    stop = 0
    data_num = 0
    MAX_STEP = 100

    # For overall metrics
    all_success = []
    all_osr = []
    all_ne = []
    all_spl = []

    # Group by environment type
    env_groups = {}
    for item in all_eval_info:
        env_type = item["image_path"].split("/")[0]  # Get environment type
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)
    
    # Process each environment type sequentially
    for env_name, eval_info in env_groups.items():
        print(f"Starting evaluation of environment: {env_name}, with {len(eval_info)} data entries")
        time.sleep(5)
        
        # Create appropriate environment bridge based on environment type
        if "airsim" in env_name:
            env_bridge = AirsimBridge(env_name)
            pos_ratio = 1.0
        elif "ue" in env_name:
            env_bridge = UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name)
            pos_ratio = 1.0
        elif "gs" in env_name:
            env_bridge = GSBridge(env_name)
            pos_ratio = 5.15
        else:
            print(f"Unknown environment type: {env_name}, skipping")
            continue
        
        # Evaluate all data for current environment
        for idx, item in enumerate(eval_info):
            acts = []  # Reset action list
            
            pos_list = item['pos']
            text = item['gpt_instruction']
            start_postion = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")
            
            stop_error = 1
            image_error = False
            
            # Set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            env_bridge.set_camera_pose(
                start_postion[0]/pos_ratio, 
                start_postion[1]/pos_ratio, 
                start_postion[2]/pos_ratio, 
                pitch, 
                np.rad2deg(start_yaw), 
                0
            )
            
            step = 0
            flag_osr = 0
            image_list = []
            # env_bridge.pass_len = 1e-3
            # env_bridge.pass_len = 10
            env_bridge.pass_len = 5
            # env_bridge.pass_len = 1
            old_pose = new_pose
            data_num += 1
            same_seq = False
            
            while step < MAX_STEP:
                try:
                    raw_image = env_bridge.get_camera_data()
                    cv2.imwrite("test/cur_img.jpg", raw_image)
                    image = raw_image
                    
                    image_list.append(image)
                    # model_action = get_action(policy, processor, image_list, text, acts, if_his=True, his_step=2)
                    model_action = get_action3(model, image_list, text, norm_stats, action_tokenizer, same_seq)
                    # model_action = get_action2(policy, image_list, text, acts, if_his=True, his_step=2)
                    # model_action = get_action3(policy, processor, image_list, text, acts, if_his=True, his_step=2)
                    acts.append(model_action)
                    same_seq = True
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    env_bridge.set_camera_pose(
                        new_pose[0]/pos_ratio, 
                        new_pose[1]/pos_ratio, 
                        new_pose[2]/pos_ratio, 
                        pitch, 
                        np.rad2deg(new_pose[3]), 
                        0
                    )
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = new_pose

                    if model_action == 0:
                        stop_error = 0
                        break
                    step += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break

            dis = calculate_distance(end_position, new_pose)
            env_bridge.traj_len = calculate_distance(end_position, start_postion)
            # env_bridge.distance_to_goal.append(dis)
            env_bridge.distance_to_goal.append(dis + 50)
            if dis < 20:
                acc += 1
                env_bridge.success.append(1)
                env_bridge.spl.append(env_bridge.traj_len / env_bridge.pass_len)
                # env_bridge.spl.append(env_bridge.pass_len / env_bridge.traj_len)
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0)
            if flag_osr == 0:
                env_bridge.osr.append(0)
            env_bridge.print_info()

            # >>> 新增：收集当前样本的指标 <<<
            all_success.append(env_bridge.success[-1])  # 最后一个样本的成功标志
            all_osr.append(env_bridge.osr[-1])  # 最后一个样本的OSR标志
            all_ne.append(env_bridge.distance_to_goal[-1])  # 最后一个样本的NE
            all_spl.append(env_bridge.spl[-1])

            if image_error:
                continue
                
        
        # Clean up environment resources
        print(f"Completed evaluation of environment {env_name}")
        kill_env_process("AirVLN")
        kill_env_process("guangzhou")
        kill_env_process("shanghai")
        kill_env_process("CitySample")
        kill_env_process("CrashReport")

        # 按 env_name 分组输出指标
        # group_sr = np.mean(env_bridge.success) if env_bridge.success else 0.0
        # group_osr = np.mean(env_bridge.osr) if env_bridge.osr else 0.0
        # group_ne = np.mean(env_bridge.distance_to_goal) if env_bridge.distance_to_goal else 0.0
        # print(f"Environment {env_name} - SR: {group_sr:.4f}, OSR: {group_osr:.4f}, NE: {group_ne:.4f}")

        del env_bridge
        import gc
        gc.collect()

        # 只跑 airsim16
        # break
    
    # Final results
    final_acc = acc / data_num if data_num > 0 else 0
    final_stop = 1 - stop / data_num if data_num > 0 else 0
    
    print(f"\nEvaluation complete!")
    print(f"Total samples: {data_num}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final stop rate: {final_stop:.4f}")

    avg_sr = np.mean(all_success) if all_success else 0.0
    avg_osr = np.mean(all_osr) if all_osr else 0.0
    avg_ne = np.mean(all_ne) if all_ne else 0.0
    avg_spl = np.mean(all_spl) if all_spl else 0.0

    # print(f"\nEvaluation complete!")
    # print(f"Total samples: {data_num}")
    # print(f"Final accuracy (deprecated): {final_acc:.4f}")
    # print(f"Final stop rate (deprecated): {final_stop:.4f}")
    print(f"---")
    print(f"Average Success Rate (SR): {avg_sr:.4f}")
    print(f"Average Oracle Success Rate (OSR): {avg_osr:.4f}")
    print(f"Average Navigation Error (NE): {avg_ne:.4f}")
    print(f"Average Split Error (SPL): {avg_spl:.4f}")


if __name__ == '__main__':
    main()
