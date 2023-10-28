from tqdm.auto import tqdm
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.import_utils import is_xformers_available
import PIL.Image
from dinov_2 import get_dinov2_model
from UNet2D import UNet2DConditionModel


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # prepare the diffusion noise scheduler for training
        self.train_noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_path, subfolder='scheduler'
        )

        # prepare the diffusion noise scheduler for evaluation(inference, sampling)
        if self.args.eval_scheduler == 'ddpm':
            self.sample_noise_scheduler = DDPMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder='scheduler')
        elif self.args.eval_scheduler == 'ddim':
            self.sample_noise_scheduler = DDIMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder='scheduler')
        else:
            self.sample_noise_scheduler = PNDMScheduler.from_pretrained(
                args.pretrained_model_path, subfolder='scheduler')
        
        # load models
        if args.image_encoder == 'clip':
            self.image_feature_extractor = CLIPImageProcessor.from_pretrained(
                args.pretrained_model_path, subfolder='feature_extractor'
            )   
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                args.pretrained_model_path, subfolder='image_encoder'
            )
            if not self.args.use_clip_proj:
                print('Train the linear layer initialized from the CLIP projection layer')
                self.refer_clip_proj = torch.nn.Linear(self.image_encoder.visual_projection.in_features, self.image_encoder.visual_projection.out_features, bias=False)
                self.refer_clip_proj.load_state_dict(self.image_encoder.visual_projection.state_dict())
                self.refer_clip_proj.requires_grad_(True)
            print('CLIP image encoder loaded')
        elif args.image_encoder == 'dinov2':
            self.image_encoder = get_dinov2_model(self.args.dinov2_path, version=self.args.dinov2_version, pretrained=False)
            self.dinov2_head = nn.Linear(1536, 768)
            self.dinov2_head.requires_grad_(True)
            print(f'DINOv2 image encoder loaded, version {self.args.dinov2_version}')
        else:
            raise NotImplementedError
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_path, subfolder='vae'
        )
        print('VAE loaded')

        
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path, subfolder='unet'
        )
        if self.args.pose_injection == 'concat_before_conv_in':
            self.unet.set_conv_in(in_channels=8)
        elif self.args.pose_injection == 'addition_after_conv_in':
            raise NotImplementedError
        else:
            raise NotImplementedError
        print('UNet loaded')

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                print("xformers is available, therefore enabled")
            else:
                print("xformers is not available, therefore not enabled")

        # other settings
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.scale_factor = self.args.scale_factor # 0.1825
        self.guidance_scale = self.args.guidance_scale
        self.controlnet_conditioning_scale = getattr(self.args, "controlnet_conditioning_scale", 1.0)

        print('Model loaded')

    def enable_vae_slicing(self):
        self.vae.enable_slicing()
    
    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def freeze_pretrained_model(self):
        '''
        Freeze partial parameters in the pretrained model which are not to be trained
        '''
        # freeze vae and image encoder
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)

        # freeze unet
        self.unet.train()
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True) # conv_in is modified
        if self.args.unet_trainable_module == 'all':
            for name, para in self.unet.named_parameters():
                para.requires_grad_(True)
            # print('UNet trainable modules: all')
        elif self.args.unet_trainable_module == 'cross_attn': # only cross attn (attn2)
            for name, para in self.unet.named_parameters():
                if 'transformer_blocks' in name and 'attn2' in name:
                    para.requires_grad_(True)
            # print('UNet trainable modules: cross attn')
        elif self.args.unet_trainable_module == 'transformer': # whole transformer
            for name, para in self.unet.named_parameters():
                if 'transformer_blocks' in name:
                    para.requires_grad_(True)
            # print('UNet trainable modules: transformer')
        elif self.args.unet_trainable_module == 'self_cross_attn': # only self & cross attn (attn1, attn2)
            for name, para in self.unet.named_parameters():
                if 'transformer_blocks' in name and ('attn1' in name or 'attn2' in name):
                    para.requires_grad_(True)
            # print('UNet trainable modules: self & cross attn')
        else:
            self.unet.requires_grad_(False)
            self.unet.eval()
            # print('UNet trainable modules: none')

    def get_trainable_parameters(self):
        '''
        return the state_dict of trainable parameters
        '''
        trainable_dict = {}
        for name, para in self.named_parameters():
            if para.requires_grad:
                trainable_dict[name] = para
        return trainable_dict

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.unet.parameters()).dtype
        self.image_encoder.float()
        return model_converted

    def half(self, *args, **kwargs):
        super().half(*args, **kwargs)
        self.dtype = torch.float16
        self.image_encoder.float()
        return
    
    def train(self, *args):
        super().train(*args)
        self.freeze_pretrained_model()

    def vae_encode_image(self, image):
        '''
        Encode a batch of images into latent space using VQ-VAE
        '''
        b, c, h, w = image.size()
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.scale_factor
        latents = latents.to(dtype=self.dtype)
        return latents
    
    def vae_decode_image(self, latents):
        ''''
        Decode a batch of latents back into images.
        '''
        latents = 1/self.scale_factor * latents
        dec = self.vae.decode(latents).sample
        image = (dec / 2 + 0.5).clamp(0, 1)
        return image
    
    def forward(self, inputs):       
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.args.seed)
        inputs['generator'] = generator

        if self.training:
            outputs = self.forward_train(inputs)
        else:
            outputs = self.forward_sample(inputs)
        removed_key = inputs.pop("generator", None)
        return outputs

    def clip_encode_image_global(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False):
        """
        Return the embedding output of the Linear layer which processes the global cls token after the ViT backbone.
        Sequence length of the output hidden states is only 1, which I think would make the attention mechanism trivial.
        """
        assert self.args.image_encoder == 'clip', 'Only support CLIP image encoder'
        dtype = next(self.image_encoder.parameters()).dtype

        # if not isinstance(image, torch.Tensor) or (image.shape[1] != 3 and image.shape[1] != 1):
        #     image = self.image_feature_extractor(images=image, return_tensors="pt").pixel_values
            
        image = image.to(device=self.device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1) # (bs, 1, 768)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.dtype)

    def clip_encode_image_all(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False):
        '''
        Return the embedding output of the Linear layer which process both the global cls token and the local patch tokens.
        The linear layer can be trainable according to args.use_clip_proj.
        '''
        assert self.args.image_encoder == 'clip', 'Only support CLIP image encoder'
        dtype = next(self.image_encoder.parameters()).dtype
        # if not isinstance(image, torch.Tensor) or (image.shape[1] != 3 and image.shape[1] != 1):
        #     image = self.image_feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=dtype)
        last_hidden_states = self.image_encoder(image).last_hidden_state # (bs, 257, 1024)
        last_hidden_states_norm = self.image_encoder.vision_model.post_layernorm(last_hidden_states)

        if self.args.use_clip_proj: # directly use clip pretrained projection layer
            image_embeddings = self.image_encoder.visual_projection(last_hidden_states_norm)
        else:
            image_embeddings = self.refer_clip_proj(last_hidden_states_norm.to(dtype=self.dtype))

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape # (bs, 257, 768)
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance: 
            if self.args.cfg_zero_image_first: # encode zero-image as condition
                uncond_imgae = torch.zeros_like(image)
                uncond_last_hidden_states = self.image_encoder(uncond_imgae).last_hidden_state
                uncond_last_hidden_states_norm = self.image_encoder.vision_model.post_layernorm(uncond_last_hidden_states)
                if self.args.use_clip_proj:
                    uncond_image_embeddings = self.clip_image_encoder.visual_projection(uncond_last_hidden_states_norm)
                else:
                    uncond_image_embeddings = self.refer_clip_proj(uncond_last_hidden_states_norm.to(dtype=self.dtype))
                bs_embed, seq_len, _ = uncond_image_embeddings.shape
                uncond_image_embeddings = uncond_image_embeddings.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = uncond_image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
            else: # simply use zero-initialized tensor as condition
                negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.dtype)

    def clip_endoer_image_multi_feature(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False):
        '''
        Return not only the final output embeddings, but also some intermediate hidden states.
        '''
        assert self.args.image_encoder == 'clip', 'Only support CLIP image encoder'
        dtype = next(self.image_encoder.parameters()).dtype
        # if not isinstance(image, torch.Tensor) or (image.shape[1] != 3 and image.shape[1] != 1):
        #     image = self.image_feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=dtype)
        outputs = self.image_encoder(image, output_hidden_states=True)
 
        hidden_states = outputs.hidden_states # 25 x (bs, 257, 1024) for 24 encoder layers + 1 for input embedding

        # TODO: Select which hidden states to use?
        selected_indices = [1, 12, 24]
        selected_hidden_states = [hidden_states[i] for i in selected_indices] 
        selected_hidden_states = torch.cat(selected_hidden_states, dim=1) # (bs, 257 * n, 1024)


        if self.args.use_clip_proj: # directly use clip pretrained projection layer
            image_embeddings = self.image_encoder.visual_projection(selected_hidden_states)
        else:
            image_embeddings = self.refer_clip_proj(selected_hidden_states.to(dtype=self.dtype))

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape # (bs, 257, 768)
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance: 
            if self.args.cfg_zero_image_first: # encode zero-image as condition
                uncond_imgae = torch.zeros_like(image)
                uncond_last_hidden_states = self.image_encoder(uncond_imgae).last_hidden_state
                uncond_last_hidden_states_norm = self.image_encoder.vision_model.post_layernorm(uncond_last_hidden_states)
                if self.args.use_clip_proj:
                    uncond_image_embeddings = self.clip_image_encoder.visual_projection(uncond_last_hidden_states_norm)
                else:
                    uncond_image_embeddings = self.refer_clip_proj(uncond_last_hidden_states_norm.to(dtype=self.dtype))
                bs_embed, seq_len, _ = uncond_image_embeddings.shape
                uncond_image_embeddings = uncond_image_embeddings.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = uncond_image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
            else: # simply use zero-initialized tensor as condition
                negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.dtype)
        
    def dinov2_encode_image(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False):
        assert self.args.image_encoder == 'dinov2', 'Only support DINOv2 image encoder'
        dtype = next(self.image_encoder.parameters()).dtype

        image = image.to(device=self.device, dtype=dtype)
        outputs = self.image_encoder(image, is_training=True)
        x_norm_clstoken = outputs['x_norm_clstoken'].unsqueeze(1) # (bs, 1, 1536)
        x_norm_patchtokens = outputs['x_norm_patchtokens']
        x_all_tokens = torch.cat([x_norm_clstoken, x_norm_patchtokens], dim=1) # (bs, 257, 1536)

        # project the output embeddings to the same dimension as CLIP(768)
        x_all_tokens = self.dinov2_head(x_all_tokens) # (bs, 257, 768)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = x_all_tokens.shape
        x_all_tokens = x_all_tokens.repeat(1, num_images_per_prompt, 1)
        x_all_tokens = x_all_tokens.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            if self.args.cfg_zero_image_first: # encode zero-image as condition
                raise NotImplementedError
            else: # simply use zero-initialized tensor as condition
                negative_prompt_embeds = torch.zeros_like(x_all_tokens)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            x_all_tokens = torch.cat([negative_prompt_embeds, x_all_tokens])

        return x_all_tokens.to(dtype=self.dtype)
    
    def sample_timesteps(self, batch_size, is_video=None):
        if hasattr(self.args, 'timestep_sampler') and self.args.timestep_sampler == 'adapted': 
            num_time_steps = self.train_noise_scheduler.config.num_train_timesteps
            expanded_range = int(num_time_steps * self.args.adapted_expand_ratio)
            adapted_range = int(self.args.adapted_seperate_ratio * num_time_steps)

            right_threshold = num_time_steps - adapted_range
            timesteps = torch.randint(low=0, high=expanded_range,size=(batch_size,)).long().to(self.device)
            # increase the probability of sampling larger timesteps for videos in [N-adapted_range, N]
            timesteps = torch.where(
                (timesteps >= right_threshold) * (is_video == True), # * means and
                torch.floor(right_threshold + (timesteps - right_threshold) / (expanded_range - right_threshold) * (num_time_steps - right_threshold)),
                timesteps
                ).long()
            # increase the probability of sampling smaller timesteps for images in [1, adapted_range]
            timesteps = torch.where(
                (timesteps >= num_time_steps) * (is_video == False),
                torch.floor((timesteps - num_time_steps) / (expanded_range - num_time_steps) * adapted_range),
                timesteps).long()
        else:
            timesteps = torch.randint(
            low=0, high=self.train_noise_scheduler.config.num_train_timesteps,
            size=(batch_size,)
            ).long().to(self.device)
        return timesteps

    def prepare_extra_step_kwargs(self, generator, eta=0.0):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.sample_noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.sample_noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
            self, batch_size, num_channels_latents,
            height, width, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], dtype=self.dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(self.device)
            else:
                latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=self.device, dtype=self.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        return latents
    
    def forward_train(self, inputs):
        image = inputs['diffusion_target_image'].to(self.device) # (bs, c, h, w)
        img_encoder_pre_ref_fg = inputs['image_encoder_preprocessed_source_image'].to(self.device)
        target_pose_image = inputs['target_pose_image'].to(self.device)
        bs = image.shape[0]

        # 1.encode the reference image for cross attention
        if self.args.ref_encoder_type == 'clip_global':
            ref_fg_emb = self.clip_encode_image_global(img_encoder_pre_ref_fg)
        elif self.args.ref_encoder_type == 'clip_all':
            ref_fg_emb = self.clip_encode_image_all(img_encoder_pre_ref_fg)
        elif self.args.ref_encoder_type == 'clip_multi':
            ref_fg_emb = self.clip_endoer_image_multi_feature(img_encoder_pre_ref_fg)
        elif self.args.ref_encoder_type == 'dinov2':
            ref_fg_emb = self.dinov2_encode_image(img_encoder_pre_ref_fg)
        else:
            raise NotImplementedError

        
        # 2.prepare the initial latents for diffusion process
        latents = self.vae_encode_image(image).to(dtype=self.dtype)
        pose_map_latents = self.vae_encode_image(target_pose_image).to(dtype=self.dtype)
        noise = torch.randn_like(latents)

        # cfg sampling
        thresholds = torch.rand(bs, device=self.device)
        thresholds = thresholds[:, None, None]
        ref_fg_emb = torch.where(thresholds >= self.args.cfg_eta, ref_fg_emb, 0.)
        thresholds = thresholds.unsqueeze(-1)
        pose_map_latents = torch.where(thresholds >= self.args.cfg_eta, pose_map_latents, 0.)

        # 3.sample the timesteps and do the forward diffusion process
        timesteps = self.sample_timesteps(bs)
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps) # pose map should not be added noise

        # 4. concat pose and noisy image
        if self.args.pose_injection == 'concat_before_conv_in':
            noisy_latents = torch.concat([noisy_latents, pose_map_latents], dim=1)
        else:
            raise NotImplementedError

        # 5. main unet
        pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=ref_fg_emb,
        ).sample

        # 6. calculate loss
        if self.train_noise_scheduler.config.prediction_type == "epsilon": # default
            target = noise
        elif self.train_noise_scheduler.config.prediction_type == "v_prediction":
            target = self.train_noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.train_noise_scheduler.config.prediction_type}")
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        outputs = dict()
        outputs['loss_total'] = loss
        return outputs
    
    @torch.no_grad()
    def forward_sample(self, inputs):
        gt_image = inputs['diffusion_target_image']
        img_encoder_pre_ref_fg = inputs['image_encoder_preprocessed_source_image'].to(self.device)
        target_pose_image = inputs['target_pose_image'].to(self.device)
        bs,_,h,w = gt_image.shape
        do_classifier_free_guidance = self.guidance_scale > 1.0
        outputs = dict()

        # 1.encode the reference image for cross attention
        if self.args.ref_encoder_type == 'clip_global':
            ref_fg_emb = self.clip_encode_image_global(img_encoder_pre_ref_fg, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)
        elif self.args.ref_encoder_type == 'clip_all':
            ref_fg_emb = self.clip_encode_image_all(img_encoder_pre_ref_fg, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)
        elif self.args.ref_encoder_type == 'clip_multi':
            ref_fg_emb = self.clip_endoer_image_multi_feature(img_encoder_pre_ref_fg, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)
        elif self.args.ref_encoder_type == 'dinov2':
            ref_fg_emb = self.dinov2_encode_image(img_encoder_pre_ref_fg, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)
        else:
            raise NotImplementedError
        

        # 2.prepare the initial latents
        num_channels_latents = self.unet.config.in_channels
        gen_height = h
        gen_width = w
        generator = inputs['generator']
        latents = self.prepare_latents(
            bs * self.args.num_inf_images_per_prompt,
            num_channels_latents,
            gen_height,
            gen_width,
            generator,
            latents=None,
        )

        # 3. concat pose and noisy image
        if self.args.pose_injection == 'concat_before_conv_in':
            pose_map_latents = self.vae_encode_image(target_pose_image).to(dtype=self.dtype)

        # 4.denoising loop
        self.sample_noise_scheduler.set_timesteps(
            self.args.num_inference_steps, device=self.device)
        timesteps = self.sample_noise_scheduler.timesteps
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)

        num_warmup_steps = len(timesteps) - self.args.num_inference_steps * self.sample_noise_scheduler.order
        with self.progress_bar(total=self.args.num_inference_steps, desc='Denoising loop') as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    # latents_input = torch.cat([latents]*2)
                    if self.args.pose_injection == 'concat_before_conv_in':
                        latents_and_pose = torch.concat([latents, pose_map_latents], dim=1)
                        latents_and_zeros = torch.concat([latents, torch.zeros_like(pose_map_latents)], dim=1)
                        latents_input = torch.concat([latents_and_pose, latents_and_zeros])
                else:
                    if self.args.pose_injection == 'concat_before_conv_in':
                        latents_input = torch.concat([latents, pose_map_latents], dim=1)
                # some schedulers that need to scale the denoising model input depending on the current timestep.
                latents_input = self.sample_noise_scheduler.scale_model_input(latents_input, t) 

                # unet
                pred = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=ref_fg_emb).sample.to(dtype=self.dtype)
                
                # cfg guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = pred.chunk(2)
                    pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = self.sample_noise_scheduler.step(pred, t, latents, **extra_step_kwargs).prev_sample # x_t -> x_{t-1}

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.sample_noise_scheduler.order == 0):
                    progress_bar.update()
        
        # 5. decode the latents
        gen_img = self.vae_decode_image(latents)
        outputs['logits_imgs'] = gen_img
        return outputs
    
    def prepare_image(
        self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image
    
    def progress_bar(self, iterable=None, total=None,desc=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config, desc=desc)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config, desc=desc)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

if __name__ == '__main__':
    # parse args
    import argparse
    from yaml import safe_load
    from accelerate.utils import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        yaml_args = safe_load(f)
    parser.set_defaults(**yaml_args)
    args = parser.parse_args()
    if not hasattr(args, 'num_iters'):
        args.num_iters = 200
    set_seed(args.seed)

    device = 'cuda:1'
    model = Net(args).to(device)
    model.train()
    print(f'UNet conv_in',model.unet.conv_in)

    from dataset import DeepFashionDataset
    from torch.utils.data import DataLoader
    dataset = DeepFashionDataset(args, 'train', args.data_root_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    for i, batch in enumerate(dataloader):
        outputs = model(batch)

        if i == 0:
            break