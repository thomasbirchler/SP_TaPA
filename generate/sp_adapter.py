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
import socket

# all directions
directions = ["left", "front", "right"]


######### MODEL #########

def prepare_model():
    # Args:
    #     prompt: The prompt/instruction (Alpaca style).
    #     input: Optional input (Alpaca style).

    # adapter_path: Path to the checkpoint with trained adapter weights, which are the output of `finetune_adapter.py`.
    # adapter_path: Path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned.pth")

    # pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
    # pretrained_path: Path = Path("llama-2-7b/consolidated.00.pth")

    # quantize: Whether to quantize the model and using which method:
    # ``"llm.int8"``: LLM.int8() mode, ``"gptq.int4"``: GPTQ 4-bit mode.
    quantize: Optional[str] = None

    # assert adapter_path.is_file()
    assert pretrained_path.is_file()

    # Initialize a 'Fabric' object for device configuration and management.
    fabric = L.Fabric(devices=1)

    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    # with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
    with lazy_load(pretrained_path) as pretrained_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        # TODO: uncomment if model is fine-tuned
        # model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    # print(f"Max_seq_len input model: {max_new_tokens}", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)
    return model


def run_model(model, prompt, iteration, max_new_tokens, top_k, temperature):
    """Generates a response based on a given instruction and an optional input.
        This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
        See `finetune_adapter.py`.
    """
    # tokenizer_path: The tokenizer path to load.
    # tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")
    tokenizer_path: Path = Path("checkpoints/llama/tokenizer.model")

    assert tokenizer_path.is_file()
    tokenizer = Tokenizer(tokenizer_path)
    # sample = {"instruction": prompt, "input": input}
    # prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    # print("===================")
    # print(prompt)
    # print("===================")
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
    save_string(prompt, iteration, "prompt")
    output = output.split("my answer is: ")[1].strip()
    save_string(output, iteration, "command")

    tokens_generated = y.size(0) - prompt_length
    print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    # if fabric.device.type == "cuda":
    #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

    return output


######### CREATE PROMPT #########

def create_prompt(iteration, target_object):
    prompt = "###### Initial Situation:\n"
    prompt += preprompt(target_object)

    prompt += "\n###### Captionized Scenes:\n"
    captionized_scenes = get_scene_caption(iteration)
    scene_prompt = generate_scene_prompt(captionized_scenes)
    prompt += scene_prompt

    prompt += "\n###### Visible Objects:\n"
    object_list = get_object_list(iteration)
    object_prompt = generate_object_prompt(object_list)
    prompt += object_prompt

    prompt += "\n###### Question:\n"
    prompt += f"Which direction is most likely to get me closer to the {target_object}?\n"

    prompt += "\n###### Answer:\n"
    prompt += f"Giving the answer with only left, front or right, my answer is: "

    save_string(prompt, iteration, "prompt")

    return prompt


def preprompt(target_object) -> str:
    preprompt = "You are a sophisticated robot which helps humans find objects within their homes. " \
                f"Your task is to tell the direction which is most likely to get you closer to the {target_object}." \
                "The objects visible to your left, front and right are all listed in the " \
                "Visible Objects section. Your answer must be one word and can only be one of these words: " \
                "Left, front, right."
    return preprompt


def wait_until_path_is_available(path):
    while True:
        if os.path.exists(path):
            break


def get_scene_caption(iteration):
    scene_caption = [None, None, None]
    counter = 0
    for direction in directions:
        # Open the file in read mode
        file_path = f"input/captions/caption{iteration:04}_{direction}.txt"
        wait_until_path_is_available(file_path)
        try:
            with open(file_path, "r") as file:
                scene_caption[counter] = file.readline()
        except Exception as e:
            print(f"An error occurred: {e}")
        counter += 1
    return scene_caption


def generate_scene_prompt(scenes):
    scene_prompt = ""
    for scene, direction in zip(scenes, directions):
        scene_prompt += f"The following sentence is the scene caption of direction {direction}: {scene}\n"
    return scene_prompt


def get_object_list(iteration):
    object_list = [None, None, None]
    dir = 0
    for direction in directions:
        # Open the file in read mode
        file_path = f"input/objects/objects{iteration:04}_{direction}.txt"
        wait_until_path_is_available(file_path)
        with open(file_path, "r") as file:
            # Read the entire file into the string, replacing newline characters with ", "
            objects = file.read().replace("\n", ", ")
            objects = objects[:-2] + "."
            object_list[dir] = objects
        dir += 1
    return object_list


def generate_object_prompt(objects):
    object_prompt = ""
    for object, direction in zip(objects, directions):
        # object_prompt += f"The following are the objects visible in direction {direction}: {object}\n"
        object_prompt += f"These objects are visible in direction {direction}: {object}\n"
    return object_prompt


######### Saving of Prompt, Command and Processed_Command #########

def save_string(string, iteration, mode, test=False, test_number=0):
    output_file = ""
    if mode == "command":
        if test:
            output_file = f"output/tests/test_{test_number:02}/command{iteration:04}.txt"
        else:
            output_file = f"output/commands/command{iteration:04}.txt"
    if mode == "prompt":
        if test:
            output_file = f"output/tests/test_{test_number:02}/prompt{iteration:04}.txt"
        else:
            output_file = f"output/prompts/prompt{iteration:04}.txt"
    if mode == "processed_command":
        if test:
            output_file = f"output/tests/test_{test_number:02}/processed_command{iteration:04}.txt"
        else:
            output_file = f"output/processed_commands/processed_command{iteration:04}.txt"

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
        # file_data = b''
        saving_folder = ""
        file_name = b''

        # Receive the filename
        while True:
            data = client_socket.recv(1)
            if data == b'\0':
                break
            file_name += data
        # Decode the filename
        file_name = file_name.decode("utf-8")

        # Exit program if target object is found
        if file_name == "exit":
            return True

        # Specify the saving location and file name
        if "caption" in file_name:
            saving_folder = "input/captions/"
        if "object" in file_name:
            saving_folder = "input/objects/"
        # saving_location = os.path.join(save_location, file_name)
        current_directory = os.getcwd()
        saving_location = os.path.join(current_directory, saving_folder, file_name)

        # Receive and save the file
        with open(saving_location, 'wb') as file:
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
                    file.write(file_data)
                    print(f"Text file {file_name} saved.")
                    return False


def send_data_to_socket(iteration, client_socket):
    file_name = f"processed_command{iteration:04}.txt"
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "output/processed_commands", file_name)
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



######### Prepare Command #########

def count_occurrences(file_path, word):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Count the occurrences of "left" (case-insensitive)
            count = content.lower().count(word)
        return count
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return 0


def process_command(iteration):
    # Specify the path to the input text file
    file_path = f'output/commands/command{iteration:04}.txt'

    # Create a dictionary to store counts
    count_dict = {
        "left": count_occurrences(file_path, "left"),
        "right": count_occurrences(file_path, "right"),
        "front": count_occurrences(file_path, "front")
    }

    # Find the variable name with the largest count
    max_count_variable = max(count_dict, key=count_dict.get)

    if max_count_variable == "left":
        if count_dict.get("left") == count_dict.get("right"):
            save_string("none", iteration, "processed_command")
        elif count_dict.get("left") == count_dict.get("front"):
            save_string("none", iteration, "processed_command")
        else:
            save_string(max_count_variable, iteration, "processed_command")
    else:
        save_string(max_count_variable, iteration, "processed_command")



######### Main #########

def main():
    # set target object which one wants to find
    target_object = "fridge"
    target_object_found = False

    # Setting parameters for LLM
    max_new_tokens = 64
    top_k = 80
    temperature = 0.35

    # Preparing model
    model = prepare_model()

    # Set-up connection with local machine and get target object
    client_socket = setup_socket()
    target_object = client_socket.recv(1024)
    target_object = target_object.decode("utf-8")

    # Keep track of which step/frame we are processing
    iteration = 0

    while True:
        # Get the data from local machine
        for i in range(6):
            target_object_found = receive_data_from_socket(client_socket)
            if target_object_found:
                break

        if target_object_found:
            break

        created_prompt = create_prompt(iteration, target_object)
        run_model(model, created_prompt, iteration, max_new_tokens, top_k, temperature)

        # Extract direction
        process_command(iteration)

        # Send command to local machine
        send_data_to_socket(iteration, client_socket)

        iteration += 1
        time.sleep(5)

    # Close socket after target has been found
    client_socket.close()


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)



######### Unused Code #########

"""
def create_input(iteration):
    # get objects and scene captioning as list. Left, front, right corresponding to index 0,1,2 respectively
    object_list = get_object_list(iteration)
    scene_caption = get_scene_caption(iteration)
    input = fuse_objects_and_scene(object_list, scene_caption)
    return input


def fuse_objects_and_scene(objects, caption) -> str:
    input = ""
    for i in range(len(directions)):
        # input = input + f"On the {directions[i]} there are following objects: {objects[i]}. With this image caption: {caption[i]}. "
        input = input + f"Objects visible on the {directions[i]}: {objects[i]}\n"
    return input


def testing_different_parameters(model, max_new_tokens_start, top_k_start, temperature_start, test_number):
    max_new_tokens = max_new_tokens_start
    for tokens in range(1):
        top_k = top_k_start
        for top in range(4):
            temperature = temperature_start
            for temp in range(3):
                iteration = 0

                output_file = f"output/test_{test_number}/parameters.txt"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                test_parameters = f"Testnumber: {test_number}, Temperature: {temperature}, top_k: {top_k}, max_new_tokens: {max_new_tokens}"
                print(test_parameters)
                with open(output_file, "w") as file:
                    file.write(test_parameters)

                for i in range(0, 4):
                    prompt = create_prompt(iteration, target_object="ButterKnife")
                    run_model(model, prompt, test_number, max_new_tokens, top_k, temperature)
                    iteration += 1
                test_number += 1
                temperature += 0.25
            top_k *= 2
        max_new_tokens *= 4
        
"""
