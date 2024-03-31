import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
from PIL import Image

from deep_utils import CRNNInferenceTorch, put_text_on_image, split_extension


def run_inference(model_ckpt: str, image_file: str, save_result: bool = False):
    recognizer = CRNNInferenceTorch(model_ckpt)
    
    pil_img = Image.open(image_file)
    
    start_time = time.time()
    text_prediction = recognizer.infer(image_file)
    text_prediction = "".join(text_prediction)
    end_time = time.time()

    if save_result:
        img_cv = cv2.imread(image_file)
        img_cv = put_text_on_image(
            img_cv, text_prediction, position=(20, 20),
            font_path="assets/Vazir.ttf", font_size=32
        )
        output_path = split_extension(image_file, suffix="_pred")
        cv2.imwrite(output_path, img_cv)

    print(f"Recognized Text: {text_prediction}")
    print(f"Processing Time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", default="output/exp_1/best.ckpt")
    parser.add_argument("--input_image", default="sample_images/image_01.jpg")
    parser.add_argument("--save_output", action="store_true")
    args = parser.parse_args()

    run_inference(
        model_ckpt=args.ckpt_path,
        image_file=args.input_image,
        save_result=args.save_output
    )
