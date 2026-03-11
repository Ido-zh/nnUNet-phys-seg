
import torch
import numpy as np 
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.network_training.nnUNetTrainerV2_InvGreAug import nnUNetTrainerV2_InvGreAug
from nnunet.training.data_augmentation.ginipa.gin import GINGroupConv
from nnunet.training.data_augmentation.ginipa.advbias import AdvBias, rescale_intensity


class nnUNetTrainerV2_InvGreRC(nnUNetTrainerV2_InvGreAug):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

    def initialize(self, training=True, force_load_plans=False):
        ret = super().initialize(training, force_load_plans)
        self.gin = GINGroupConv(out_channel=1, in_channel=1, interm_channel=2, n_layer=4)
        blender_cofig = {
            'epsilon': 0.3,
            'xi': 1e-6,
            'control_point_spacing':[24, 24],
            'downscale':2, #
            'data_size':[self.batch_size, 1, *self.patch_size],
            'interpolation_order':2,
            'init_mode':'gaussian',
            'space':'log'
        }
        self.ipa =  AdvBias(blender_cofig)
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        self.gin.cuda().float()
        self.ipa.cuda().float()

        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)


        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        gin1, gin2 = [self.gin(data) for _ in range(2)]

        # generate correlation maps (bias field)
        self.ipa.init_parameters()
        blend_mask = rescale_intensity(self.ipa.bias_field)
        tau1 = gin1*blend_mask + gin2*(1-blend_mask)
        tau2 = gin2*blend_mask + gin1*(1-blend_mask)
        data = tau1 if np.random.random() < 0.5 else tau2 

        self.optimizer.zero_grad()

        if self.fp16:
            with torch.amp.autocast('cuda'):
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()