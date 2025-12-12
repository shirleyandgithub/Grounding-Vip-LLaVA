# encoding=utf-8

import re
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, VipLlavaForConditionalGeneration, BitsAndBytesConfig

LOCAL_MODEL_PATH = "/root/vip_llava"
MARKED_IMAGE_PATH = "/root/vip_llava.png"
OUTPUT_IMAGE = "/root/result.png"

# 模型推理超参
MAX_NEW_TOKENS = 50  # 最大生成token数，坐标输出仅需少量字符，避免冗余计算
DO_SAMPLE = False    # 关闭随机采样，确保输出结果可复现（确定性生成）
PAD_TOKEN_ID = None  # 填充token ID，后续从processor中动态获取模型对应的eos_token_id
NUM_BEAMS = 3        # 搜索数量，平衡精度与速度（增大可提升召回率，但增加耗时）
REPETITION_PENALTY = 1.5  # 重复惩罚系数：抑制模型输出重复内容，避免坐标冗余

PROMPT = "精确检测图中穿红裤、蓝色针织帽、黑色上衣的男性。输出格式：[x1,y1,x2,y2]（归一化坐标，仅保留数字，精确到小数点后4位）"

# 后处理超参：控制目标框的像素级紧贴效果，适配红裤、蓝帽、黑上衣的特征分布
MARGIN = 3  # 最终框的边界宽松度（像素），平衡紧贴效果与特征完整性
RED_PANTS_LOWER_H_RATIO = 0.55  # 红裤检测区域起始比例（ROI高度的55%处开始），聚焦下半身特征
HEAD_H_RATIO = 0.35  # 头部检测区域高度比例（ROI上35%区域），限定蓝帽搜索范围
HEAD_TOP_MARGIN = 8  # 蓝帽顶部边界宽松度（像素），避免框体截断头部
RED_BOTTOM_MARGIN = 12  # 红裤底部边界宽松度（像素），确保完整框选红裤区域

# 量化配置超参
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4位量化，降低显存占用
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 像素级紧贴收缩函数
def shrink_to_person(bbox, img_cv):
    h, w = img_cv.shape[:2]  # 获取图像的高度和宽度
    x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]
    roi = img_cv[y1:y2, x1:x2]
    if roi.size == 0:
        return (x1, y1, x2, y2)

    # 红裤定位
    lower_h = int(roi.shape[0] * RED_PANTS_LOWER_H_RATIO)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv[lower_h:], (0, 90, 60), (10, 255, 255)) + \
               cv2.inRange(hsv[lower_h:], (160, 90, 60), (180, 255, 255))
    red_y, red_x = np.where(red_mask > 0)
    if len(red_x) == 0:
        return (x1, y1, x2, y2)

    # 蓝帽定位
    head_h = int(roi.shape[0] * HEAD_H_RATIO)
    head = roi[:head_h, red_x.min():red_x.max()]
    hsv_head = cv2.cvtColor(head, cv2.COLOR_BGR2HSV) if head.size > 0 else None
    blue_mask = cv2.inRange(hsv_head, (105, 80, 60), (125, 255, 255)) if hsv_head is not None else None
    by = np.where(blue_mask > 0)[0] if blue_mask is not None and len(blue_mask) > 0 else []
    head_top = by.min() if len(by) > 0 else 0

    # 最终紧贴框
    final_x1 = x1 + max(red_x.min() - MARGIN, 0)
    final_x2 = x1 + min(red_x.max() + MARGIN, roi.shape[1])
    final_y1 = y1 + max(head_top - HEAD_TOP_MARGIN, 0)
    final_y2 = y1 + min(lower_h + red_y.max() + RED_BOTTOM_MARGIN, roi.shape[0])

    return (max(0, final_x1), max(0, final_y1), min(w, final_x2), min(h, final_y2))

def main():
    image = Image.open(MARKED_IMAGE_PATH).convert("RGB")

    # 加载模型与处理器
    model = VipLlavaForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=QUANTIZATION_CONFIG,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False
    )
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = processor.tokenizer.eos_token_id

    # 构建输入与推理
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=PAD_TOKEN_ID,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
        )

    # 解析与后处理
    result = processor.decode(outputs[0], skip_special_tokens=True).split("###Assistant:")[-1].strip()
    bbox = [float(x) for x in re.findall(r'\d+\.?\d*', result)[:4]]\

    img_cv = cv2.imread(MARKED_IMAGE_PATH)
    final_bbox = shrink_to_person(bbox, img_cv)

    # 绘制Grounding主框
    cv2.rectangle(img_cv, final_bbox[:2], final_bbox[2:], (0, 255, 0), 4)
    cv2.putText(img_cv, "Grounding",
                (final_bbox[0], final_bbox[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

    cv2.imwrite(OUTPUT_IMAGE, img_cv)
    print(f"归一化坐标：[{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}]")
    print(f"结果保存至：{OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()



