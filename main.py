import yaml
import argparse
import image_processing as img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', default='images')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--n-clusters', type=int)
    parser.add_argument('--mask-threshold', type=float)
    parser.add_argument('--single', metavar='FILE')
    parser.add_argument('--purge', action='store_true')
    parser.add_argument('--purge-all', action='store_true')
    args = parser.parse_args()

    # CLI overrides win over yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.n_clusters:
        config['n_clusters'] = args.n_clusters
    if args.mask_threshold:
        config['mask_threshold'] = args.mask_threshold
    
    if args.purge_all:
        img.purge_images(purge_all=True)
    elif args.purge:
        img.purge_images()
    elif args.single:
        img.process_single_image(args.single, config)
    else:
        img.process_all_images(args.image_dir, args.output_dir)

