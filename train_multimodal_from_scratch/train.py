from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any


class VLMConfig(PretrainedConfig):
    """
    VLMConfig类用于存储多模态模型的配置参数。
    包含视觉模型路径、语言模型路径、是否冻结视觉模型参数、图像填充数量等配置。
    """
    model_type = "vlm_model"

    def __init__(self, llm_model_path='/home/user/Downloads/Qwen2.5-0.5B-Instruct',
                 vision_model_path='/home/user/Downloads/siglip-so400m-patch14-384',
                 freeze_vision_model=True,
                 image_pad_num=49,
                 **kwargs):
        # 初始化配置参数
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    """
    VLM类实现了一个多模态模型，结合视觉和语言模型。
    通过线性层将视觉特征映射到语言模型的输入维度。
    """
    config_class = VLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 加载视觉模型和处理器
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        # 加载语言模型和分词器
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        # 定义线性层用于特征映射
        # 因为后面图像嵌入被调整为每个token的特征维度是原来的4倍（d*4）。因此，线性层的输入维度*4
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size * 4,
                                 self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        # 冻结视觉模型参数（如果配置中指定）
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        # 冻结语言模型参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        # 获取文本嵌入
        # self.llm_model.get_input_embeddings() 返回的是一个嵌入层对象（通常是 nn.Embedding），这个对象是可调用的。
        # (input_ids) 是对这个嵌入层对象的调用，传入 input_ids 以获取嵌入向量。
        # 嵌入层（nn.Embedding）的工作原理可以被视为一种“查表”操作
        # 嵌入层的核心是一个嵌入矩阵，形状为 (vocab_size, embedding_dim)，这个矩阵的每一行对应词汇表中的一个词的嵌入向量。
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        # 获取图像嵌入
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        # 其中b是批量大小，s是序列长度（即图像tokens的数量），d是每个token的特征维度。
        b, s, d = image_embeds.shape
        # 压缩图像tokens
        # 将每4个tokens的特征拼接在一起，形成一个新的token。每个token的特征维度变为d*4。
        # 因为attention操作的计算复杂度与序列长度的平方成正比，因此压缩到49，可以显著降低计算复杂度。
        image_embeds = image_embeds.view(b, -1, d * 4)  # (b, 196, d) --> (b, 49, d*4)
        # 使用两个线性层对压缩后的图像特征进行映射，转换到与语言模型输入相匹配的维度。使用SiLU激活函数
        # # (b, 49, d*4) --> (b, 49, H) llm_model.config.hidden_size（记为 H，即语言模型的隐藏向量维度）。
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        # 确保文本嵌入和图像特征的数据类型一致
        text_embeds = text_embeds.to(image_features.dtype)
        # 合并文本和图像特征
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        # 通过语言模型进行前向传播
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            # 计算损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        # 合并输入ID和图像特征
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, iamge_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        inputs_embeds[batch_indices, iamge_indices] = image_features.view(-1, embed_dim)
        return inputs_embeds


class MyDataset(Dataset):
    """
    MyDataset类用于加载和处理训练数据。
    从JSON文件中加载对话数据，并处理图像。
    """

    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        # 加载数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # 获取样本数据
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            # 处理问题文本
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": conversations[0]['value']}], \
                                                        tokenize=False, \
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            # 处理答案文本
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            # 将文本转换为token id
            # input_id是文本中每个词或子词在词汇表（vocabulary）中的索引。
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            # 标签labels用于训练时的目标输出。对于问题部分，标签用pad_token_id填充，因为模型不需要预测问题部分的内容
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            # 去掉input_ids的最后一个元素，这样做是因为最后一个词不需要预测下一个词。
            # 去掉labels的第一个元素，这样做是因为第一个词没有前面的词来预测它。
            # 去掉input_ids的最后一个元素和labels的第一个元素是为了实现这种对齐，以便模型能够学习预测下一个词。
            # 例如：输入: "The cat is on the"；标签: "cat is on the mat"
            # 这样，模型在每个时间步都能根据输入的前面部分预测下一个词，并将预测结果与标签进行比较，计算损失。
            input_ids = input_ids[:-1]
            labels = labels[1:]
            # 处理图像
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            # 处理异常情况
            # 如果无法加载指定的图像，代码会创建一个白色的默认图像（224x224像素）。
            # 生成一个默认的用户问题文本，询问“图片内容是什么”
            # 生成一个默认的回答文本“图片内容为空”。
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role": "system", "content": 'You are a helpful assistant.'},
                                                         {"role": "user", "content": "图片内容是什么\n<image>"}], \
                                                        tokenize=False, \
                                                        add_generation_prompt=True).replace('<image>',
                                                                                            '<|image_pad|>' * self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }


class MyDataCollator:
    """
    MyDataCollator类用于整理批量数据，确保输入序列和标签的长度一致。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 整理批量数据
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            # 填充输入和标签
            # 对于每个feature，将input_ids和labels填充到max_len，确保所有输入和标签的长度一致。
            input_ids.append(
                feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            # 将图像特征添加到列表中
            # 因为图像数据通常已经是固定大小的张量，所以不需要填充。
            pixel_values.append(feature['pixel_values'])

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}


if __name__ == '__main__':
    # 初始化配置和模型
    config = VLMConfig(vision_model_path='/home/user/wyf/siglip-base-patch16-224', image_pad_num=49)
    # 创建VLM模型实例，并将其移动到GPU上
    model = VLM(config).cuda()
    # 打印模型结构和可训练参数数量
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 设置数据路径
    images_path = './dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain'
    # 设置训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    # 初始化Trainer并开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()

