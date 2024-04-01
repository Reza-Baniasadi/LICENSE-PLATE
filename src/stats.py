from argparse import ArgumentParser
from dataset import get_mean_std
from get_optimum_img_w import get_optimum_img_w
from get_character_sets import get_unique_characters


def main_stats():
    arg_parser = ArgumentParser(description="Compute dataset statistics for OCR")
    arg_parser.add_argument("--dataset_path", required=True, type=str, help="Directory containing dataset")
    arg_parser.add_argument("--batch_sz", default=128, type=int, help="Batch size for DataLoader")
    arg_parser.add_argument("--image_height", default=32, type=int, help="Target image height")

    options = arg_parser.parse_args()

    char_set = get_unique_characters(options.dataset_path)
    longest_length, best_width = get_optimum_img_w(options.dataset_path, char_set)
    mean_vals, std_vals = get_mean_std(
        options.dataset_path,
        char_set,
        options.batch_sz,
        options.image_height,
        best_width
    )

    print(
        f"[DATASET INFO] Characters: {char_set}, Optimal width: {best_width}, "
        f"Mean: {mean_vals}, Std: {std_vals}, "
        f"Num classes: {len(char_set)+1}, Max label length: {longest_length}"
    )


if __name__ == "__main__":
    main_stats()
