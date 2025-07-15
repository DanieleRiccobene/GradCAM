import os
import argparse
import torch
import glob
from tqdm import tqdm

from net import SUM, load_and_preprocess_image, predict_saliency_map, overlay_heatmap_on_image, write_heatmap_to_image
from net.configs.config_setting import setting_config


def setup_model(device):
    config = setting_config
    model_cfg = config.model_config
    if config.network == 'sum':
        model = SUM(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
        )
        model.load_state_dict(torch.load('net/pre_trained_weights/sum_model.pth', map_location=device))
        model.to(device)
        return model
    else:
        raise NotImplementedError("The specified network configuration is not supported.")


def process_dataset(dataset_path, output_path, model, device, condition_map, heat_map_type):
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        condition = condition_map.get(class_name, 0)
        output_class_dir = os.path.join(output_path, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        image_paths = glob.glob(os.path.join(class_dir, '*.*'))
        for image_path in tqdm(image_paths, desc=f"Processing {class_name}"):
            try:
                img, orig_size = load_and_preprocess_image(image_path)
                pred_saliency = predict_saliency_map(img, condition, model, device)

                filename = os.path.splitext(os.path.basename(image_path))[0]
                hot_output_filename = os.path.join(output_class_dir, f'{filename}_saliencymap.png')

                write_heatmap_to_image(pred_saliency, orig_size, hot_output_filename)

                if heat_map_type == 'Overlay':
                    overlay_output_filename = os.path.join(output_class_dir, f'{filename}_overlay.png')
                    overlay_heatmap_on_image(image_path, hot_output_filename, overlay_output_filename)
            except Exception as e:
                print(f"Errore con {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Batch Saliency Map Prediction')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='results_dataset')
    parser.add_argument('--heat_map_type', type=str, default='HOT', choices=['HOT', 'Overlay'])
    parser.add_argument('--from_pretrained', type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.from_pretrained:
        model = SUM.from_pretrained(args.from_pretrained).to(device)
    else:
        model = setup_model(device)
    model.eval()

    # Mapping nome classe → condition (modifica se serve)
    condition_map = {
        'n02056570': 1,
        'n02085936': 1,
        'n02690373': 1
    }

    process_dataset(args.dataset_path, args.output_path, model, device, condition_map, args.heat_map_type)


if __name__ == "__main__":
    main()
