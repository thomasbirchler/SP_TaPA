import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt
# from sp_socket  import setup_socket, receive_and_save_from_socket
import socket

# all directions
directions = ["left", "front", "right"]


######### MODEL #########
def prepare_model():
    # Args:
    #     prompt: The prompt/instruction (Alpaca style).
    #     input: Optional input (Alpaca style).

    # adapter_path: Path to the checkpoint with trained adapter weights, which are the output of `finetune_adapter.py`.
    adapter_path: Path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned.pth")
    # pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
    # quantize: Whether to quantize the model and using which method:
    # ``"llm.int8"``: LLM.int8() mode, ``"gptq.int4"``: GPTQ 4-bit mode.
    quantize: Optional[str] = None

    assert adapter_path.is_file()
    assert pretrained_path.is_file()

    # Initialize a 'Fabric' object for device configuration and management.
    fabric = L.Fabric(devices=1)

    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        # model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    # print(f"Max_seq_len input model: {max_new_tokens}", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)
    return model


def run_model(model, prompt, test_number, iteration, input=""):
    """Generates a response based on a given instruction and an optional input.
        This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
        See `finetune_adapter.py`.
    """
    # tokenizer_path: The tokenizer path to load.
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")
    # max_new_tokens: The number of generation steps to take.\
    max_new_tokens: int = 100
    # top_k: The number of top most probable tokens to consider in the sampling process.
    top_k: int = 200
    # temperature: A value controlling the randomness of the sampling process. Higher values result in more random samples.
    temperature: float = 0.8

    assert tokenizer_path.is_file()
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    print("===================")
    print(prompt)
    print("===================")
    print(f"token shape: {encoded.shape}", file=sys.stderr)
    prompt_length = encoded.size(0)

    t0 = time.perf_counter()
    y = generate(
        model,
        idx=encoded,
        max_seq_length=max_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id
    )
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    save_string(prompt, iteration, test_number, command=False, test=True)
    save_string(output, iteration, test_number, command=True, test=True)
    output = output.split("### Response:")[1].strip()
    print(output)

    tokens_generated = y.size(0) - prompt_length
    print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    # if fabric.device.type == "cuda":
    #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

    return output


######### PREPARE PROMPT AND INPUT #########

def wait_until_path_is_available(path):
    while True:
        if os.path.exists(path):
            break


def get_object_list(iteration):
    object_list = [None, None, None]
    dir = 0
    for direction in directions:
        # Open the file in read mode
        file_path = f"input/objects/frame{iteration:04}_{direction}.txt"
        wait_until_path_is_available(file_path)
        with open(file_path, "r") as file:
            # Read the entire file into the string, replacing newline characters with ", "
            objects = file.read().replace("\n", ", ")
            objects = objects[:-2] + "."
            object_list[dir] = objects
        dir += 1
    return object_list


def get_scene_caption(iteration):
    scene_caption = [None, None, None]
    counter = 0
    for direction in directions:
        # Open the file in read mode
        file_path = f"input/captions/caption{iteration:04}_{direction}_prompt3.txt"
        wait_until_path_is_available(file_path)
        try:
            with open(file_path, "r") as file:
                scene_caption[counter] = file.readline()
        except Exception as e:
            print(f"An error occurred: {e}")
        counter += 1
    return scene_caption


def fuse_objects_and_scene(objects, caption) -> str:
    input = ""
    for i in range(len(directions)):
        # input = input + f"On the {directions[i]} there are following objects: {objects[i]}. With this image caption: {caption[i]}. "
        input = input + f"Objects visible on the {directions[i]}: {objects[i]}\n"
    return input


def create_input(iteration):
    # get object list and scene captioning.
    object_list = get_object_list(iteration)
    scene_caption = get_scene_caption(iteration)
    input = fuse_objects_and_scene(object_list, scene_caption)
    # return "[Bread, DiningTable, Egg, Drawer, Toaster, Fork, Potato, Mirror, GarbageBag, AluminumFoil, Sink, Plate,
    # Cup, CounterTop, SoapBottle, Shelf, Chair, StoveKnob, Pan, ButterKnife, CoffeeMachine, PepperShaker, Spoon,
    # Pot, Window, LightSwitch, Cabinet, Spatula, SaltShaker, Apple, Faucet, StoveBurner, GarbageCan, Bowl, Lettuce,
    # Fridge, Knife, Microwave, Mug, Tomato, Blinds, DishSponge, SideTable]"
    return input


def create_prompt(iteration, target_object):
    prompt = preprompt(target_object)

    prompt += "Captionized Scenes:\n"
    captionized_scenes = get_scene_caption(iteration)
    scene_prompt = generate_scene_prompt(captionized_scenes)
    prompt += scene_prompt

    prompt += "Visible Objects:\n"
    object_list = get_object_list(iteration)
    object_prompt = generate_object_prompt(object_list)
    prompt += object_prompt

    prompt += "Instruction: \n"
    prompt += f"You want to help come closer to the {target_object} within one step towards either left, front " \
              "or right. For this I go one step towards the "

    return prompt


def generate_object_prompt(objects):
    object_prompt = ""
    for object, direction in zip(objects, directions):
        object_prompt += f"The following are the objects visible in direction {direction}: {object}"
    return object_prompt


def generate_scene_prompt(scenes):
    scene_prompt = ""
    for scene, direction in zip(scenes, directions):
        scene_prompt += f"The following sentence is the scene caption of direction {direction}: {scene}"
    return scene_prompt


def preprompt(target_object) -> str:
    preprompt = "You are a sophisticated robot which helps humans find objects within their homes. The three " \
                "different directions (left, front, right) are described for you in words and the visible objects are " \
                "listed. Your task is to predict the direction (left, front or right) which has the highest " \
                f"probability to lead to the target object: {target_object}. Always answer with only one of these words:" \
                " \"left\", \"front\" or \"right\".\n"

    return preprompt


######### Saving of Prompt or Command #########

def save_string(string, iteration, test_number, command=True, test=False):
    output_file = ""
    if command:
        if test:
            output_file = f"output/test_{test_number:02}/command{iteration}.txt"
        else:
            output_file = f"output/command{iteration}.txt"
    else:
        if test:
            output_file = f"output/test_{test_number:02}/prompt{iteration}.txt"
        else:
            output_file = f"output/prompt{iteration}.txt"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        file.write(string)


######### SOCKET #########
def setup_socket():
    host = '192.168.1.204'
    port = 54320
    #  Create a server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(4)
    print("Waiting for a connection...")

    # Accept a client connection
    client_socket, addr = server_socket.accept()
    print("Connected by", addr)

    return client_socket


def receive_data_from_socket(client_socket):
    while True:
        file_data = b''
        file_name = b''

        # Receive the filename
        while True:
            data = client_socket.recv(1)
            if data == b'\0':
                break
            file_name += data
        # Decode the filename
        file_name = file_name.decode("utf-8")

        # Specify the saving location and file name
        # save_location = "/home/bthomas/SP_TaPA/input/socket/"
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, file_name)

        # Receive and save the file based on its extension
        with open(file_path, 'wb') as file:
            buffer = b''
            while True:
                # Receive and save the text file
                data = client_socket.recv(1024)
                if not data:
                    break
                # Add the newly received data to the buffer
                buffer += data
                while b'FILE_END' in buffer:
                    # Split buffer at the first occurrence of 'FILE_END'
                    file_data, buffer = buffer.split(b'FILE_END', 1)
                    # Write the data to the file
                    # file_data = file_data.decode("utf-8")
                    file.write(file_data)
                    print(f"Text file {file_name} saved.")
                    return


def send_data_to_socket(iteration, client_socket):
    file_name = f"command{iteration:04}.txt"
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "output", file_name)
    # Send the filename first
    client_socket.send(file_name.encode("utf-8"))

    # Send a delimiter to separate filename and content
    client_socket.send(b'\0')
    with open(file_path, 'rb') as text_file:
        # Send file in batches
        text_data = text_file.read(1024)
        while text_data:
            client_socket.send(text_data)
            text_data = text_file.read(1024)
    # Send a marker to indicate the end of the file
    client_socket.send(b'FILE_END')
    print(f"File {file_name} sent.")


def main():
    # client_socket = setup_socket()
    # receive_data_from_socket(client_socket)
    # send_data_to_socket(iteration, "left", client_socket)

    test_number = 13

    # prompt = ["In which direction should I go to find the fridge?",
    #           "Should I go left, front or right to find the fridge?",
    #           "You are a robot which is build to help humans. I want you to find the fridge. In which direction do you go?",
    #           "You are a robot which is build to help humans. I want you to find the fridge. Do you go left, front or rigth?"]

    model = prepare_model()

    for iteration in range(0, 3):
        prompt = create_prompt(iteration, target_object="Fridge")
        navigation_command = run_model(model, prompt, test_number, iteration)


    # for p in range(len(prompt)):
    #     iteration = 0
    #     created_input = create_input(iteration)
    #     print(f"Prompt inserted into the model. Now prompt = {p}.")
    #     navigation_command = run_model(model, prompt[p], test_number, iteration, created_input)
    #     save_string(navigation_command, test_number, "command", iteration)
        # test_number += 1


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
    # client_socket = setup_socket()
    # for i in range(6):
    #     receive_data_from_socket(client_socket, iteration)
    # send_data_to_socket(0, client_socket)
