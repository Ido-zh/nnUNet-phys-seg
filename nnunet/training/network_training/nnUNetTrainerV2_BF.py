import torch 
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2, maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.ginipa.advbias import AdvBias, rescale_intensity


class nnUNetTrainerV2_BF(nnUNetTrainerV2):
    def initialize(self, training=True, force_load_plans=False):
        ret = super().initialize(training, force_load_plans)
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
        self.batch_size = 20 
        self.max_num_epochs = 500
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
        self.ipa.cuda().float()

        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)


        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        # generate correlation maps (bias field)
        self.ipa.init_parameters()
        blend_mask = rescale_intensity(self.ipa.bias_field)
        data = data * blend_mask

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