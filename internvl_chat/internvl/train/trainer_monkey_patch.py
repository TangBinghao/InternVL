import json
import os

from peft import PeftModel
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
logger = logging.get_logger(__name__)


def get_num_layer_for_vit_and_qllama(var_name, vit_num_max_layer, llama_num_max_layer):
    if var_name.startswith('internvl.'):
        var_name = var_name[len('internvl.'):]
    if var_name in ('query_tokens', 'logit_scale',):
        return 0
    if var_name.startswith('clip_projector.'):
        return vit_num_max_layer
    if var_name.startswith('clip_projector2.') or var_name.startswith('itm_head.') or \
            var_name == 'text_projection':
        return llama_num_max_layer
    if var_name.startswith('vision_model.'):
        if 'embeddings.' in var_name:
            return 0
        if 'layers.' in var_name:
            var_name = var_name.split('layers.')[-1]
            layer_id = int(var_name.split('.')[0])
            return layer_id + 1
    if var_name.startswith('qllama.'):
        if 'embed_tokens' in var_name:
            return 0
        if 'layers.' in var_name:
            var_name = var_name.split('layers.')[-1]
            layer_id = int(var_name.split('.')[0])
            return layer_id + 1
        else:
            return llama_num_max_layer
    return 0


def param_classification(name):
    if name.startswith('internvl.'):
        name = name[len('internvl.'):]
    if name in ['query_tokens', 'text_projection', 'logit_scale']:
        return 'qllama'
    elif name.startswith('vision_model.'):
        return 'vit'
    elif name.startswith('qllama.'):
        return 'qllama'
    elif name.startswith('clip_projector.'):
        return 'vit'
    elif name.startswith('clip_projector2.'):
        return 'qllama'
    elif name.startswith('itm_head.'):
        return 'qllama'
    else:
        return 'other'


def create_optimizer(self):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    parameter_groups = {}
    try:  # for stage2 model
        vit_num_layers = opt_model.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.config.qllama_config.num_hidden_layers + 2
    except:  # for stage3 model
        vit_num_layers = opt_model.internvl.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.internvl.config.qllama_config.num_hidden_layers + 2
    print('vit_num_layers:', vit_num_layers)
    print('qllama_num_layers:', qllama_num_layers)

    vit_layer_decay_rate = float(os.getenv('VIT_LAYER_DECAY_RATE', 1.0))
    qllama_layer_decay_rate = float(os.getenv('QLLAMA_LAYER_DECAY_RATE', 1.0))
    qllama_lr_scale = float(os.getenv('QLLAMA_LR_SCALE', 1.0))
    print('vit_layer_decay_rate:', vit_layer_decay_rate)
    print('qllama_layer_decay_rate:', qllama_layer_decay_rate)
    print('qllama_lr_scale:', qllama_lr_scale)

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = self.args.weight_decay

        cls = param_classification(name)
        layer_id = get_num_layer_for_vit_and_qllama(name, vit_num_layers, qllama_num_layers)
        group_name = '%s_layer_%d_%s' % (cls, layer_id, group_name)
        if group_name not in parameter_groups:
            if cls == 'vit':
                scale = vit_layer_decay_rate ** (vit_num_layers - layer_id - 1)
            elif cls == 'qllama':
                scale = qllama_layer_decay_rate ** (qllama_num_layers - layer_id - 1)
                scale = scale * qllama_lr_scale
            else:
                scale = 1.0
            scale = min(1.0, scale)
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * self.args.learning_rate,
            }
        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

        rank = torch.distributed.get_rank()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print('Param groups = %s' % json.dumps(to_display, indent=2))

    optimizer_grouped_parameters = list(parameter_groups.values())
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == 'Adam8bit':
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                logger.info(f'skipped {module}: {skipped / 2 ** 20}M params')
                manager.register_module_override(module, 'weight', {'optim_bits': 32})
                logger.debug(f'bitsandbytes: will optimize {module} in fp32')
        logger.info(f'skipped: {skipped / 2 ** 20}M params')

    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        self.optimizer = smp.DistributedOptimizer(self.optimizer)

    return self.optimizer


def replace_create_optimizer():
    print('Replace original create_optimizer with custom create_optimizer')
    transformers.Trainer.create_optimizer = create_optimizer

def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    @staticmethod
    def compute_margin_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, pos_labels: torch.Tensor, neg_labels: torch.Tensor, tokenizer, margin: float = 0.2) -> torch.Tensor:
        # logits: bsz, seq_len, vocab_size
        # labels: bsz, seq_len
        label_ids = [x[-1] for x in tokenizer(['0', '1']).input_ids]
        level0, level1 = label_ids
        # Shift labels to align with logits
        shift_pos_labels = pos_labels[..., 1:]
        shift_neg_labels = neg_labels[..., 1:]
        
        # Get the scores for the token_id at the label positions
        pos_scores_level1 = pos_logits[..., :-1, level1]
        neg_scores_level1 = neg_logits[..., :-1, level1]
        pos_scores_level0 = pos_logits[..., :-1, level0]
        neg_scores_level0 = neg_logits[..., :-1, level0]
        
        # Mask to ignore padding tokens
        pos_masks = shift_pos_labels != -100
        neg_masks = shift_neg_labels != -100
        pos_scores_level1 = pos_scores_level1[pos_masks]
        neg_scores_level1 = neg_scores_level1[neg_masks]
        pos_scores_level0 = pos_scores_level0[pos_masks]
        neg_scores_level0 = neg_scores_level0[neg_masks]
        
        # Compute margin loss for level1
        # TODO: mismatch for the pos_scores_level1 - neg_scores_level1, e.g., pos_scores_level1 size: torch.Size([0]) neg_scores_level1 size: torch.Size([2])
        margin_loss_level1 = torch.clamp(margin - (pos_scores_level1 - neg_scores_level1), min=0.0)
        # Compute margin loss for level0
        margin_loss_level0 = torch.clamp(margin - (neg_scores_level0 - pos_scores_level0), min=0.0)
        # Combine the two margin losses
        total_margin_loss = margin_loss_level1.mean() + margin_loss_level0.mean()
        return total_margin_loss
    pos_inputs = inputs['pos']
    neg_inputs = inputs['neg']
    pos_labels, neg_labels = None, None
    if self.label_smoother is not None and 'pos_labels' in inputs:
        pos_labels = inputs['pos'].pop('pos_labels')
    if self.label_smoother is not None and 'neg_labels' in inputs:
        neg_labels = inputs['neg'].pop('neg_labels')
    
    # outputs = model(**inputs)
    pos_outputs = model(**pos_inputs)
    neg_outputs = model(**neg_inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if pos_labels is not None:
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            pos_loss = self.label_smoother(pos_outputs, pos_labels, shift_labels=True)
        else:
            pos_loss = self.label_smoother(pos_outputs, pos_labels)
    else:
        pos_loss = pos_outputs['loss'] if isinstance(pos_outputs, dict) else pos_outputs[0]
    
    if neg_labels is not None:
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            neg_loss = self.label_smoother(neg_outputs, neg_labels, shift_labels=True)
        else:
            neg_loss = self.label_smoother(neg_outputs, neg_labels)
    else:
        neg_loss = neg_outputs['loss'] if isinstance(neg_outputs, dict) else neg_outputs[0]
    # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    if pos_labels is None:
        pos_labels = pos_inputs['labels']
    if neg_labels is None:
        neg_labels = neg_inputs['labels']
    margin_loss = compute_margin_loss(pos_outputs.logits, neg_outputs.logits, pos_labels, neg_labels, self.tokenizer)
    # print('margin_loss',margin_loss)
    total_loss = pos_loss + neg_loss + margin_loss
    if return_outputs:
        outputs = {
            'pos_outputs': pos_outputs,
            'neg_outputs': neg_outputs,
            'pos_loss': pos_loss,
            'neg_loss': neg_loss,
            'margin_loss': margin_loss,
            'total_loss': total_loss
        }
        return total_loss, outputs
    else:
        return total_loss

def replace_compute_loss():
    print('Replace original compute_loss with custom compute_loss for pairwise')
    transformers.Trainer.compute_loss = compute_loss


from transformers import Trainer as HfTrainer

class MyTrainer(HfTrainer):
    @staticmethod
    def compute_margin_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, pos_labels: torch.Tensor, neg_labels: torch.Tensor, tokenizer, margin: float = 0.2) -> torch.Tensor:
        # logits: bsz, seq_len, vocab_size
        # labels: bsz, seq_len
        label_ids = [x[-1] for x in tokenizer(['0', '1']).input_ids]
        level0, level1 = label_ids
        # Shift labels to align with logits
        shift_pos_labels = pos_labels[..., 1:]
        shift_neg_labels = neg_labels[..., 1:]
        
        # Get the scores for the token_id at the label positions
        pos_scores_level1 = pos_logits[..., :-1, level1]
        neg_scores_level1 = neg_logits[..., :-1, level1]
        pos_scores_level0 = pos_logits[..., :-1, level0]
        neg_scores_level0 = neg_logits[..., :-1, level0]
        
        # Mask to ignore padding tokens
        pos_masks = shift_pos_labels != -100
        neg_masks = shift_neg_labels != -100
        pos_scores_level1 = pos_scores_level1[pos_masks]
        neg_scores_level1 = neg_scores_level1[neg_masks]
        pos_scores_level0 = pos_scores_level0[pos_masks]
        neg_scores_level0 = neg_scores_level0[neg_masks]
        
        # Compute margin loss for level1
        # TODO: mismatch for the pos_scores_level1 - neg_scores_level1, e.g., pos_scores_level1 size: torch.Size([0]) neg_scores_level1 size: torch.Size([2])
        try: 
            margin_loss_level1 = torch.clamp(margin - (pos_scores_level1 - neg_scores_level1), min=0.0)
        except:
            margin_loss_level1 = torch.zeros(pos_scores_level1.shape)
        # Compute margin loss for level0
        try:
            margin_loss_level0 = torch.clamp(margin - (neg_scores_level0 - pos_scores_level0), min=0.0)
        except:
            margin_loss_level0 = torch.zeros(pos_scores_level1.shape)
        
        # Combine the two margin losses
        total_margin_loss = margin_loss_level1.mean() + margin_loss_level0.mean()
        return total_margin_loss
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        pos_inputs = inputs['pos']
        neg_inputs = inputs['neg']
        pos_labels, neg_labels = None, None
        if self.label_smoother is not None and 'pos_labels' in inputs:
            pos_labels = inputs['pos'].pop('pos_labels')
        if self.label_smoother is not None and 'neg_labels' in inputs:
            neg_labels = inputs['neg'].pop('neg_labels')
        
        # outputs = model(**inputs)
        pos_outputs = model(**pos_inputs)
        neg_outputs = model(**neg_inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if pos_labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                pos_loss = self.label_smoother(pos_outputs, pos_labels, shift_labels=True)
            else:
                pos_loss = self.label_smoother(pos_outputs, pos_labels)
        else:
            pos_loss = pos_outputs['loss'] if isinstance(pos_outputs, dict) else pos_outputs[0]
        
        if neg_labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                neg_loss = self.label_smoother(neg_outputs, neg_labels, shift_labels=True)
            else:
                neg_loss = self.label_smoother(neg_outputs, neg_labels)
        else:
            neg_loss = neg_outputs['loss'] if isinstance(neg_outputs, dict) else neg_outputs[0]
        # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if pos_labels is None:
            pos_labels = pos_inputs['labels']
        if neg_labels is None:
            neg_labels = neg_inputs['labels']
        margin_loss = self.compute_margin_loss(pos_outputs.logits, neg_outputs.logits, pos_labels, neg_labels, self.tokenizer)
        # print('margin_loss',margin_loss)
        total_loss = pos_loss + neg_loss + margin_loss
        if return_outputs:
            outputs = {
                'pos_outputs': pos_outputs,
                'neg_outputs': neg_outputs,
                'pos_loss': pos_loss,
                'neg_loss': neg_loss,
                'margin_loss': margin_loss,
                'total_loss': total_loss
            }
            return total_loss, outputs
        else:
            return total_loss