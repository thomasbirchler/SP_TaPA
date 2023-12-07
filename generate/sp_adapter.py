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


# Global path variables
# adapter_path: Path to the checkpoint with trained adapter weights, which are the output of `finetune_adapter.py`.
adapter_path: Path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned.pth")
# pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
# tokenizer_path: The tokenizer path to load.
tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")
# Initialize a 'Fabric' object for device configuration and management.
fabric = L.Fabric(devices=1)
# Keep track which frame we are at
frame_number = 0
# all directions
directions = ["left", "front", "right"]

# Global model parameters
# quantize: Whether to quantize the model and using which method:
# ``"llm.int8"``: LLM.int8() mode, ``"gptq.int4"``: GPTQ 4-bit mode.
quantize: Optional[str] = None
# max_new_tokens: The number of generation steps to take.\
max_new_tokens: int = 100
# top_k: The number of top most probable tokens to consider in the sampling process.
top_k: int = 200
# temperature: A value controlling the randomness of the sampling process. Higher values result in more random samples.
temperature: float = 0.4


def prepare_model():
    # Args:
    #     prompt: The prompt/instruction (Alpaca style).
    #     input: Optional input (Alpaca style).
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

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
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    print(f"Max_seq_len input model: {max_new_tokens}", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)
    return model


def run_model(model, prompt, input):
    """Generates a response based on a given instruction and an optional input.
        This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
        See `finetune_adapter.py`.
    """
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
    output = output.split("### Response:")[1].strip()
    print(output)

    tokens_generated = y.size(0) - prompt_length
    print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

    return output


def get_object_list():
    object_list = [None, None, None]
    counter = 0
    for direction in directions:
        # Open the file in read mode
        file_path = f"input/objects/frame{frame_number:04}_{direction}.txt"
        check_if_path_is_available(file_path)
        with open(file_path, "r") as file:
            # Read the entire file into the string, replacing newline characters with ", "
            objects = file.read().replace("\n", ", ")
            object_list[counter] = objects
        counter += 1
    return object_list


def check_if_path_is_available(path):
    while True:
        if os.path.exists(path):
            break


def get_scene_caption():
    scene_caption = [None, None, None]
    counter = 0
    for direction in directions:
        # Open the file in read mode
        file_path = f"input/captions/caption{frame_number:04}_{direction}_prompt3.txt"
        check_if_path_is_available(file_path)
        try:
            with open(file_path, "r") as file:
                scene_caption[counter] = file.readline()
        except Exception as e:
            print(f"An error occurred: {e}")
        counter += 1
    return scene_caption


def fuse_objects_and_scene(objects, caption) -> str:
    prompt = ""
    for i in range(len(directions)):
        prompt = prompt + f"On the {directions[i]} there are following objects: {objects[i]}. With this image caption: {caption[i]}. "
    return prompt


def create_input():
    # get object list and scene captioning.
    # object_list = get_object_list()
    # scene_caption = get_scene_caption()
    # inputt = fuse_objects_and_scene(object_list, scene_caption)
    return "[Bread, DiningTable, Egg, Drawer, Toaster, Fork, Potato, Mirror, GarbageBag, AluminumFoil, Sink, Plate, Cup, CounterTop, SoapBottle, Shelf, Chair, StoveKnob, Pan, ButterKnife, CoffeeMachine, PepperShaker, Spoon, Pot, Window, LightSwitch, Cabinet, Spatula, SaltShaker, Apple, Faucet, StoveBurner, GarbageCan, Bowl, Lettuce, Fridge, Knife, Microwave, Mug, Tomato, Blinds, DishSponge, SideTable]"
    # return inputt


def save_string(string, test_number, command_or_prompt):
    output_file = ""
    if command_or_prompt == "command":
        output_file = f"output/test_{test_number:02}/command{frame_number}.txt"
    elif command_or_prompt == "prompt":
        output_file = f"output/test_{test_number:02}/prompt.txt"
    else:
        print("Could not save string.")
        return
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        file.write(string)


def main():
    global frame_number
    test_number = 16
    for ii in range(1):
        prompt = "Can you make me a sandwich?"

        save_string(prompt, test_number, "prompt")
        model = prepare_model()
        for i in range(1):
            created_input = create_input()
            print(f"Prompt inserted into the model. Now i = {i} and ii = {ii}.")
            navigation_command = run_model(model, prompt, created_input)
            save_string(navigation_command, test_number, "command")
            frame_number += 1
        test_number += 1
        frame_number = 0


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
