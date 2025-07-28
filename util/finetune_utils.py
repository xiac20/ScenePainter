import torch
from tqdm import tqdm
from kornia.morphology import dilation
from torchvision.transforms import ToPILImage
import itertools
from diffusers.optimization import get_scheduler   
import pdb


def finetune_decoder(config, model, render_output, inpaint_output, n_steps=100):
    params = [{"params": model.vae.decoder.parameters(), "lr": config["decoder_learning_rate"]}]
    optimizer = torch.optim.Adam(params)
    decoder_ft_mask = render_output["inpaint_mask"].detach()
    ToPILImage()(decoder_ft_mask[0]).save(model.run_dir / 'images' / 'decoder_ft_mask.png')
    if config['dilate_mask_decoder_ft'] > 1:
        decoder_ft_mask_dilated = dilation(decoder_ft_mask, torch.ones(config['dilate_mask_decoder_ft'], config['dilate_mask_decoder_ft']).to('cuda'))
    else:
        decoder_ft_mask_dilated = decoder_ft_mask
    ToPILImage()(decoder_ft_mask_dilated[0]).save(model.run_dir / 'images' / 'decoder_ft_mask_dilated.png')
    for _ in tqdm(range(n_steps), leave=False):
        optimizer.zero_grad()
        loss = model.finetune_decoder_step(
            inpaint_output["inpainted_image"].detach(),
            inpaint_output["latent"].detach(),
            render_output["rendered_image"].detach(),
            decoder_ft_mask,
            decoder_ft_mask_dilated,
        )
        loss.backward()
        optimizer.step()

    del optimizer
    
    
def finetune_customization(config, model, render_output, inpaint_output, n_steps=100): 
    params = (itertools.chain(model.bld.text_encoder.parameters()))
    optimizer = torch.optim.AdamW(
        params,
        lr=config["customization_learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08, 
    ) 
    lr_scheduler = get_scheduler(
        'constant',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_steps,
        num_cycles=1,
        power=1,
    )
    
    # mask = get_mask(render_output["rendered_image"], object_id)
    mask = torch.ones_like(inpaint_output["inpainted_image"]).to(inpaint_output["inpainted_image"].device)
    for _ in tqdm(range(n_steps), leave=False):
        optimizer.zero_grad()
        new_recon = model.inpaint(render_output["rendered_image"], render_output["inpaint_mask"], prompt = 'a photo of <asset0> and <asset1> and <asset2> and <asset3> and <asset4>')
        loss = model.finetune_customization_step(
            inpaint_output["inpainted_image"].detach(),
            new_recon["inpainted_image"],
            mask, 
            # inpaint_output["latent"].detach(),
            # render_output["rendered_image"].detach(),
            # decoder_ft_mask,
            # decoder_ft_mask_dilated,
        )
        loss.backward()
        optimizer.step()

    del optimizer    


def finetune_depth_model(config, model, target_depth, epoch, mask_align=None, mask_cutoff=None, cutoff_depth=None):
    params = [{"params": model.depth_model.parameters(), "lr": config["depth_model_learning_rate"]}]
    optimizer = torch.optim.Adam(params)

    if mask_align is None:
        mask_align = target_depth > 0

    for _ in tqdm(range(config["num_finetune_depth_model_steps"]), leave=False):
        optimizer.zero_grad()

        loss = model.finetune_depth_model_step(
            target_depth,
            model.images[epoch],
            mask_align=mask_align,
            mask_cutoff=mask_cutoff,
            cutoff_depth=cutoff_depth,
        )
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            print('No valid pixels to compute depth fine-tuning loss. Skip this step.')
            return
