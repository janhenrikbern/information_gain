"""
CODE from https://github.com/mworchel/svbrdf-estimation

ADD CREDITS / REFERENCE BEFORE CODE RELEASE!!!
"""

import torch
import torch.nn as nn
import utils

from pytorch_svbrdf_renderer import Renderer, Config


class SVBRDFL1Loss(nn.Module):
    def forward(self, input, target):
        # Split the SVBRDF into its individual components
        (
            input_normals,
            input_diffuse,
            input_roughness,
            input_specular,
        ) = utils.unpack_svbrdf(input)
        (
            target_normals,
            target_diffuse,
            target_roughness,
            target_specular,
        ) = utils.unpack_svbrdf(target)

        epsilon_l1 = 0.01
        input_diffuse = torch.log(input_diffuse + epsilon_l1)
        input_specular = torch.log(input_specular + epsilon_l1)
        target_diffuse = torch.log(target_diffuse + epsilon_l1)
        target_specular = torch.log(target_specular + epsilon_l1)

        return (
            nn.functional.l1_loss(input_normals, target_normals)
            + nn.functional.l1_loss(input_diffuse, target_diffuse)
            + nn.functional.l1_loss(input_roughness, target_roughness)
            + nn.functional.l1_loss(input_specular, target_specular)
        )


class RenderingLoss(nn.Module):
    def __init__(self):
        super(RenderingLoss, self).__init__()

        self.renderer = Renderer()
        self.random_configuration_count = 3
        self.specular_configuration_count = 6

    def forward(self, input, target):
        views_r, lights_r, intenisities_r = utils.generate_random_scenes(self.random_configuration_count)
        views_s, lights_s, intenisities_s = utils.generate_specular_scenes(self.specular_configuration_count)

        config = Config()
        config.view_position(torch.cat([views_r, views_s], dim=0))
        config.light_position(torch.cat([lights_r, lights_s], dim=0))
        config.light_color_intensity(torch.cat([intenisities_r, intenisities_s], dim=0))
        self.renderer.set_config(config)
        input_renderings = self.renderer.run(input, stacked_channels=True)
        target_renderings = self.renderer.run(target, stacked_channels=True)
        input_renderings = input_renderings.permute(0, 1, 4, 2, 3)
        target_renderings = target_renderings.permute(0, 1, 4, 2, 3)
        batch_input_renderings = input_renderings.view(input_renderings.shape[0] * input_renderings.shape[1], *input_renderings.shape[2:])
        batch_target_renderings = target_renderings.view(target_renderings.shape[0] * target_renderings.shape[1], *target_renderings.shape[2:])

        epsilon_render = 0.1
        batch_input_renderings_logged = torch.log(
            batch_input_renderings + epsilon_render
        )
        batch_target_renderings_logged = torch.log(
            batch_target_renderings + epsilon_render
        )

        loss = nn.functional.l1_loss(
            batch_input_renderings_logged, batch_target_renderings_logged
        )

        return loss


class MixedLoss(nn.Module):
    def __init__(self, l1_weight=0.1):
        super(MixedLoss, self).__init__()

        self.l1_weight = l1_weight
        self.l1_loss = SVBRDFL1Loss()
        self.rendering_loss = RenderingLoss()

    def forward(self, input, target):
        return self.l1_weight * self.l1_loss(input, target) + self.rendering_loss(
            input, target
        )
