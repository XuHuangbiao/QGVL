import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.ARCH
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ViFi_CLIP',
                      "vision_depth": cfg.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.N_CTX_VISION,
                      "language_ctx": cfg.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(torch.cuda.FloatTensor)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = cfg.PROMPT_MODEL
        ctx_init = cfg.CTX_INIT
        ZS_evaluation = cfg.ZS_EVAL
        if ZS_evaluation:
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert cfg.PROMPT_DEPTH_TEXT >= 1, "In VL prompting, Language prompt depth should be >=1" \
                                                                 "\nPlease use VPT trainer if you want to learn only vision " \
                                                                 "branch  "
            n_ctx = cfg.N_CTX_TEXT
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            print(f"V-L design")
            print(f'Initial text context: "{prompt_prefix}"')
            print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
            print(f"Number of context words (tokens) for Vision prompting: {cfg.N_CTX_VISION}")
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            # No prompting
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings

        return prompts


class ViFiCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image1, image2):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()

        # b = image.shape[0]
        # Lets encode the video into required format
        # b, t, c, h, w = image.size()
        # Remove the batch dimensions
        # image = image.reshape(-1, c, h, w)
        # print(image.type(self.dtype).shape)
        # print(self.dtype)
        # Now pass the image into CLIP visual encoder
        # image_features = self.image_encoder(image.type(torch.cuda.FloatTensor))
        # Now again attach the batch dimensions
        # image_features = image_features.view(b, t, -1)  # [B, T, 512]
        # Now take the mean along the temporal direction
        # image_features = image_features.mean(dim=1, keepdim=False)  # image features are now ready

        # Finally, make the text features
        text_features = self.text_encoder(prompts, tokenized_prompts)

        if image1 is None:
            return text_features
        text_grade = text_features[-4:]
        text_score = text_features[:-4]
        img1 = image1 / image1.norm(dim=-1, keepdim=True)
        img2 = image2 / image2.norm(dim=-1, keepdim=True)
        # print("3:", image_features.shape)
        text_grade = text_grade / text_grade.norm(dim=-1, keepdim=True)
        text_score = text_score / text_score.norm(dim=-1, keepdim=True)
        logits1 = logit_scale * img1 @ text_grade.t()
        logits2 = logit_scale * img2 @ text_score.t()
        # print("4:", logits.shape)
        return logits1, logits2


def returnCLIP(config, class_names=None):
    print(f"Loading CLIP (backbone: {config.ARCH})")
    clip_model = load_clip_to_cpu(config)

    print("Building ViFi-CLIP CLIP")
    model = ViFiCLIP(config, class_names, clip_model)

    if config.pretrained_clip_weight:
        state_dict = torch.load(config.pretrained_clip_weight, map_location='cpu')["model"]
        new_pretrained_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_pretrained_dict.pop('prompt_learner.complete_text_embeddings')
        model.load_state_dict(new_pretrained_dict, strict=False)

    if config.PROMPT_MODEL:
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    else:
        # Now need to control freezing of CLIP for fine-tuning
        train_complete_clip = config.USE
        if train_complete_clip == "both":
            print("Turning on gradients for COMPLETE ViFi-CLIP model")
            for name, param in model.named_parameters():
                param.requires_grad_(True)
        else:
            if train_complete_clip == "image":
                print("Turning on gradients for image side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "image_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            else:
                print("Turning on gradients for TEXT side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
    # # Double check
    # enabled = set()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         enabled.add(name)
    # print(f"Parameters to be updated: {enabled}")
    # print(f"Total learnable items: {len(enabled)}")
    model.float()
    return model
