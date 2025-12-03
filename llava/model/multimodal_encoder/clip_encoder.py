import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        print("***")
        print("select layer: ", self.select_layer)
        print("***")
        select_multi_layer = getattr(args, 'select_multi_layer', '[12,16,22,23]')
        self.select_multi_layer = eval(select_multi_layer)
        print("***")
        print("select multi layer", self.select_multi_layer)
        print("***")
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        print("load vision tower done", self.vision_tower_name)
        self.is_loaded = True

    
    def feature_select(self, image_forward_outs, layers):
        ## the defalut layer index of Tokenpacker is [12,16,22,23]
        image_feature_list = []
        for l in layers:
            image_feature_list.append(image_forward_outs.hidden_states[l])

        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            image_features_multi = [x[:,1:] for x in image_feature_list]

        elif self.select_feature == 'cls_patch':
            image_features = image_features
            image_features_multi = image_feature_list
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, image_features_multi
        

    @torch.no_grad()
    def forward(self, images):

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature, image_feature_multi = self.feature_select(image_forward_out, layers=self.select_multi_layer)

                image_features.append(image_feature.to(image.dtype))
                image_features_multi.append(image_feature_multi.to(image.dtype))

        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features, image_features_multi = self.feature_select(image_forward_outs, layers=self.select_multi_layer)

        if isinstance(image_features_multi, list):
            image_features_multi = [x.to(images.dtype) for x in image_features_multi]
        #return (image_features.to(images.dtype), image_features_multi.to(images.dtype))
        return (image_features.to(images.dtype), image_features_multi)
    

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
