import os
from pathlib import Path
import model
import torch
import argparse
from eval.stability import evaluate_stability
from eval.consistency import evaluate_consistency
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', default='CUB2011', type=str)
    parser.add_argument('--data_path', type=str, default='datasets/cub200_cropped/')
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
    ppnet.load_state_dict(checkpoint)

    ppnet.to(device)
    ppnet.eval()

    consistency_score = evaluate_consistency(ppnet, args, save_dir=output_path.as_posix())
    print('Consistency Score : {:.2f}%'.format(consistency_score))
    with open(output_path / filename, 'a') as fp:
        fp.write('Consistency Score : {:.2f}%\n'.format(consistency_score))

    stability_score = evaluate_stability(ppnet, args)
    print('Stability Score : {:.2f}%'.format(stability_score))
    with open(output_path / filename, 'a') as fp:
        fp.write('Stability Score : {:.2f}%\n'.format(stability_score))
