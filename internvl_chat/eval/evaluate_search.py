import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

import zipfile
from io import BytesIO
ds_collections = {
    'wxgSearch': {
        'root': '',
        'annotation': '/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_test_mllm_swift_v2.jsonl',
        'max_new_tokens': 30,
        'min_new_tokens': 1,
    },
}

class TBHLocalDataset(torch.utils.data.Dataset):

    def __init__(self, name, root, annotation, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        # if name == 'coco':
        #     self.images = json.load(open(annotation))
        # else:
        #     self.images = json.load(open(annotation))['images']
        self.data = open(annotation,'r').readlines()
        self.name = name
        self.root = root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
    
    def load_image(self, img_path):
        try:
            if isinstance(img_path, str):
                img_path = img_path.strip()
                filename = os.path.dirname(img_path)
                img_filename = os.path.basename(img_path)
                if filename.endswith(".zip"):
                    # if not os.path.exists(filename):
                    #     raise FileNotFoundError(f"Zip file {filename} does not exist.")
                    with zipfile.ZipFile(filename, 'r') as zip_file:                        
                        img_bytes = zip_file.read(img_filename)
                        # print(Image.open(BytesIO(img_bytes)).size)
                        image = Image.open(BytesIO(img_bytes))
                else:
                    image = Image.open(img_path)
            else:
                image = img_path
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except (zipfile.BadZipFile, FileNotFoundError) as e:
            # print(f"Error opening zip file {filename}: {e}")
            # image = torch.zeros(3, 448, 448)
            image = Image.new('RGB', (448, 448))
            # image = Image.new('RGB', (224, 224), (255, 255, 255))
        return image
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data_item = json.loads(self.data[idx])
        image_file = data_item['pre_images'] + data_item['search_images']
        num_images = len(image_file)
        input_text = "<image>\n"*num_images + data_item['query']
        gold_response = data_item['response']
        total_images = []
        num_patches_list = []
        for image_path in image_file:
            # image = Image.open(image_path)
            image = self.load_image(image_path)
            if self.dynamic_image_size:
                images = dynamic_preprocess(image, image_size=self.input_size,
                                            use_thumbnail=self.use_thumbnail,
                                            max_num=self.max_num)
            else:
                images = [image]
            total_images += images
            num_patches_list.append(len(images))
        pixel_values = [self.transform(image) for image in total_images]
        pixel_values = torch.stack(pixel_values)


        return {
            'input_text': input_text,
            'gold_response': gold_response,
            'pixel_values': pixel_values,
            'num_patches_list':num_patches_list,
            'searchid': data_item['searchid'],
            'pre_feedid': data_item['pre_feedid'],
            'is_person': data_item['is_person']
        }
# class CaptionDataset(torch.utils.data.Dataset):

#     def __init__(self, name, root, annotation, prompt, input_size=224, dynamic_image_size=False,
#                  use_thumbnail=False, max_num=6):
#         if name == 'coco':
#             self.images = json.load(open(annotation))
#         else:
#             self.images = json.load(open(annotation))['images']
#         self.name = name
#         self.prompt = prompt
#         self.root = root
#         self.input_size = input_size
#         self.dynamic_image_size = dynamic_image_size
#         self.use_thumbnail = use_thumbnail
#         self.max_num = max_num
#         self.transform = build_transform(is_train=False, input_size=input_size)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         if self.name == 'coco':
#             filename = self.images[idx]['image']
#             image_id = int(filename.split('_')[-1].replace('.jpg', ''))
#             image_path = os.path.join(self.root, filename)
#         else:
#             image_id = self.images[idx]['id']
#             if 'file_name' in self.images[idx]:
#                 image_path = os.path.join(self.root, self.images[idx]['file_name'])
#             else:
#                 image_path = os.path.join(self.root, self.images[idx]['image'])

#         image = Image.open(image_path)
#         if self.dynamic_image_size:
#             images = dynamic_preprocess(image, image_size=self.input_size,
#                                         use_thumbnail=self.use_thumbnail,
#                                         max_num=self.max_num)
#         else:
#             images = [image]
#         pixel_values = [self.transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)

#         return {
#             'image_id': image_id,
#             'input_text': self.prompt,
#             'pixel_values': pixel_values
#         }


def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    num_patches_list = [_['num_patches_list'] for _ in inputs]
    inputs_text = [_['input_text'] for _ in inputs]
    golds = [_['gold_response'] for _ in inputs]
    searchids = [_['searchid'] for _ in inputs]
    pre_feedids = [_['pre_feedid'] for _ in inputs]
    is_persons = [_['is_person'] for _ in inputs]
    return pixel_values, num_patches_list, inputs_text, golds, searchids, pre_feedids, is_persons


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    random.seed(args.seed)
    summaries = []
    
    for ds_name in args.datasets:
        # if torch.distributed.get_rank() == 0:
        #     results_file = f'{ds_name}_test.jsonl'
        #     results_file = os.path.join(args.out_dir, results_file)
        #     fw = open(results_file, 'w')
        
        annotation = ds_collections[ds_name]['annotation']
        if type(annotation) == list:
            annotation = annotation[0]
        dataset = TBHLocalDataset(
            name=ds_name,
            root=ds_collections[ds_name]['root'],
            annotation=annotation,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        golds, preds, search_info, inputs_text, get_outputs = [], [], [], [], []
        for idx, (pixel_values, num_patches_list, input_text, gold, searchid, pre_feedid, is_person) in tqdm(enumerate(dataloader)):
            # if idx % 500 == 0 and torch.distributed.get_rank() == 0:
            #     print(f"Having processed {idx} samples!")
            # print(input_text,gold)
            # pixel_values = pixel_values.to(torch.bfloat16).cuda()
            pixel_values = pixel_values.to(torch.bfloat16).npu()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred, more_info = model.chat_tbh(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=input_text[0],
                num_patches_list=num_patches_list[0],
                generation_config=generation_config,
                verbose=False
            )
            # print(pred, more_info)
            inputs_text.extend([input_text[0]])
            preds.extend([pred])
            search_info.extend([more_info])
            golds.extend([gold[0]])
            output_dict = {
                    'query': input_text[0],
                    'response': pred,
                    'label': gold[0],
                    'searchid': searchid[0],
                    'pre_feedid': pre_feedid[0],
                    'is_person': is_person[0]
                }
            # print(output_dict)
            output_dict.update(more_info)
            # print(output_dict)
            get_outputs.extend([output_dict])
            # if torch.distributed.get_rank() == 0:
            #     fw.write(json.dumps(output_dict,ensure_ascii=False)+'\n')

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_inputs = [None for _ in range(world_size)]
        merged_preds = [None for _ in range(world_size)]
        merged_search_info = [None for _ in range(world_size)]
        merged_golds = [None for _ in range(world_size)]
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_inputs, inputs_text)
        torch.distributed.all_gather_object(merged_preds, preds)
        torch.distributed.all_gather_object(merged_search_info, search_info)
        torch.distributed.all_gather_object(merged_golds, golds)
        torch.distributed.all_gather_object(merged_outputs, get_outputs)
        

        merged_inputs = [_ for _ in itertools.chain.from_iterable(merged_inputs)]
        merged_preds = [_ for _ in itertools.chain.from_iterable(merged_preds)]
        merged_search_info = [_ for _ in itertools.chain.from_iterable(merged_search_info)]
        merged_golds = [_ for _ in itertools.chain.from_iterable(merged_golds)]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            results_file = os.path.join(args.out_dir, results_file)
            fw = open(results_file, 'w')
            results = []
            
            for query, pred, search_info, gold, output in zip(merged_inputs, merged_preds, merged_search_info, merged_golds, merged_outputs):
                # print("debug",query, pred, search_info, gold)
                # output = {
                #     'query': query,
                #     'response': pred,
                #     'label': gold
                # }.update(search_info)
                results.append(output)
                fw.write(json.dumps(output,ensure_ascii=False)+'\n')

            # annotation = ds_collections[ds_name]['annotation']
            # if type(annotation) == list:
            #     annotation = annotation[-1]
            # coco = COCO(annotation)
            # coco_result = coco.loadRes(results_file)
            # coco_eval = COCOEvalCap(coco, coco_result)
            # coco_eval.evaluate()

            # summary = coco_eval.eval.items()
            # print(summary)
            # summaries.append([args.checkpoint, ds_name, average_length, summary])

        torch.distributed.barrier()

    # out_path = '_'.join(args.checkpoint.split('/')[-2:])
    # writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    # print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    # for summary in summaries:
    #     print(summary)
    #     writer.write(f'{summary}\n')
    # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', type=str, default='/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/output_internvl2_2b/10w_pair_dataset_internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full_pairwise/checkpoint-800')
    parser.add_argument('--checkpoint', type=str, default='/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/output_internvl2_2b_pretrained/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full_pairwise')
    parser.add_argument('--datasets', type=str, default='wxgSearch')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/infer_results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=1)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     world_size=int(os.getenv('WORLD_SIZE', '1')),
    #     rank=int(os.getenv('RANK', '0')),
    # )
    # torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    # if args.auto:
    #     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.distributed.init_process_group(
        backend='hccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.npu.set_device(int(os.getenv('LOCAL_RANK', 0)))
    if args.auto:
        os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
    
    kwargs = {'device_map': 'auto'} if args.auto else {}

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        # model = model.cuda()
        model = model.npu()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
