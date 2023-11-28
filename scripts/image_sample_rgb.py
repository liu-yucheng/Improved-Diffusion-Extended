"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from PIL import Image

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    sample_index = 0
    red = 0
    green = 0
    blue = 0
    color_incr = 0.2

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # Begin - Inject RGB noise.

        shape = args.batch_size, 3, args.image_size, args.image_size
        device = next(model.parameters()).device
        noise = th.randn(*shape, device=device)
        prev_red = red
        prev_green = green
        prev_blue = blue

        for noise_sub in noise:
            noise_sub[0].add_(-1 + 2 * red)
            noise_sub[1].add_(-1 + 2 * green)
            noise_sub[2].add_(-1 + 2 * blue)

            blue += color_incr

            if blue > 1.001:
                green += color_incr
                blue = 0.0
            # end if

            if green > 1.001:
                red += color_incr
                green = 0.0
                blue = 0.0
            # end if

            if red > 1.001:
                red = 0.0
                green = 0.0
                blue = 0.0
            # end if
        # end for

        # End - Inject RGB noise.

        batch = sample_fn(
            model,
            shape,
            noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=device
        )
        batch = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.contiguous()

        gathered_samples = [th.zeros_like(batch) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, batch)  # gather not supported with NCCL
        numpy_samples = [sample.cpu().numpy() for sample in gathered_samples]
        all_images.extend(numpy_samples)
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        # print(numpy_samples[0].shape)

        for batch in numpy_samples:
            for item in batch:
                image = Image.fromarray(item)

                # Begin - Save RGB samples.

                out_path = os.path.join(
                    logger.get_dir(),
                    f"sample_{sample_index}_red_{prev_red:.1f}_green_{prev_green:.1f}_blue_{prev_blue:.1f}.png"
                )

                prev_blue += color_incr

                if prev_blue > 1.001:
                    prev_green += color_incr
                    prev_blue = 0.0
                # end if

                if prev_green > 1.001:
                    prev_red += color_incr
                    prev_green = 0.0
                    prev_blue = 0.0
                # end if

                if prev_red > 1.001:
                    prev_red = 0.0
                    prev_green = 0.0
                    prev_blue = 0.0
                # end if

                # End - Save RGB samples.

                logger.log(f"saving to {out_path}")
                image.save(out_path, "PNG")
                sample_index += 1

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=216,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
