import os
from pathlib import Path
import model
import torch
import logging
import sys
import argparse
from eval.stability import evaluate_stability
from eval.consistency import evaluate_consistency
from eval.distinctiveness import evaluate_distinctiveness
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', default='CUB2011', type=str)
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--nb_classes', type=int, default=200)
    parser.add_argument('--test_batch_size', type=int, default=30)

    # Model
    parser.add_argument('--base_architecture', type=str, default='dinov2_vitb_exp')  # dinov2_vitb_exp
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--num_prototypes', type=int, default=2000)

    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    output_path = Path(f'outputs/{args.base_architecture}-{args.num_prototypes}')
    output_path.mkdir(parents=True, exist_ok=True)
    filename = 'eval_results.txt'

    img_size = args.input_size
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    base_architecture = args.base_architecture
    prototype_shape = [args.num_prototypes, 64, 1, 1]
    num_classes = 200
    prototype_activation_function = 'log'
    add_on_layers_type = 'regular'

    # Load the model
    ppnet = model.construct_TesNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

    ppnet = checkpoint

    log_dir = Path(args.resume).parents[1]

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "evaluate_distinctiveness.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    ppnet.to(device)
    ppnet.eval()


    evaluate_distinctiveness(ppnet, save_path=log_dir, device=device)
