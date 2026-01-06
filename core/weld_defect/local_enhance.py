import json
from gradio_client import Client, handle_file
import shutil
from tqdm import tqdm
import os
import random
import torch
from PIL import Image
import argparse
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

def generate_prompts_replacement(target_count, with_replacement=True, use_weights=True):
    """
    生成焊接缺陷提示词
    
    Args:
        target_count: 需要生成的提示词数量
        with_replacement: 是否允许有放回抽样（重复组合）
        use_weights: 是否使用权重抽样
    """
    base_prompt = "WELDSCALLOPS，在焊缝灰度图，位于焊道{pos}{loc}（收弧处）的位置添加一个凹坑缺陷： \ 
                   凹坑{dir}且轮廓呈{shape}，凹坑区域凹下去的梯度{deg}，\
                   凹坑仅出现在焊道上，灰度与背景适配。 "

    positions = ["上半部分", "下半部分"]
    locations = ["末端", "靠近末端"]
    directions = ["竖直", "水平", "倾斜"]
    shapes = ["扁圆形", "椭圆形", "不规则形状", "缝隙状"]
    degrees = [
        "明显（边缘到中心灰度骤降）且无反光（呈深黑色）",
        "平缓（边缘到中心灰度渐变）且有反光（灰色）"
    ]
    
    # 权重设置
    pos_weights = [0.5, 0.5]  # 上下部分权重
    loc_weights = [0.5, 0.5]  # 位置权重
    dir_weights = [0.3, 0.3, 0.4]  # 方向权重（倾斜更常见）
    shape_weights = [0.4, 0.3, 0.2, 0.1]  # 形态权重
    deg_weights = [0.6, 0.4]  # 梯度权重（明显更常见）
    
    prompts = []
    
    if with_replacement:
        # 有放回抽样：可以直接生成，允许重复
        for _ in range(target_count):
            if use_weights:
                # 带权重抽样
                pos = random.choices(positions, weights=pos_weights, k=1)[0]
                loc = random.choices(locations, weights=loc_weights, k=1)[0]
                dir = random.choices(directions, weights=dir_weights, k=1)[0]
                shape = random.choices(shapes, weights=shape_weights, k=1)[0]
                deg = random.choices(degrees, weights=deg_weights, k=1)[0]
            else:
                # 均匀随机抽样
                pos = random.choice(positions)
                loc = random.choice(locations)
                dir = random.choice(directions)
                shape = random.choice(shapes)
                deg = random.choice(degrees)
            
            # 生成最终prompt
            prompt = base_prompt.format(pos=pos, loc=loc, dir=dir, shape=shape, deg=deg)
            prompts.append(prompt)
            
        print(f"已生成{len(prompts)}个提示词（有放回抽样）")
        
    else:
        # 无放回抽样：保持原有逻辑，但不重复
        combinations = set()
        max_combos = len(positions) * len(locations) * len(directions) * len(shapes) * len(degrees)
        actual_target = min(target_count, max_combos)
        
        while len(prompts) < actual_target:
            if use_weights:
                # 带权重抽样
                pos = random.choices(positions, weights=pos_weights, k=1)[0]
                loc = random.choices(locations, weights=loc_weights, k=1)[0]
                dir = random.choices(directions, weights=dir_weights, k=1)[0]
                shape = random.choices(shapes, weights=shape_weights, k=1)[0]
                deg = random.choices(degrees, weights=deg_weights, k=1)[0]
            else:
                # 均匀随机抽样
                pos = random.choice(positions)
                loc = random.choice(locations)
                dir = random.choice(directions)
                shape = random.choice(shapes)
                deg = random.choice(degrees)
            
            # 去重键
            combo_key = f"{pos}_{loc}_{dir}_{shape}_{deg}"
            if combo_key in combinations:
                continue
            
            # 生成最终prompt
            prompt = base_prompt.format(pos=pos, loc=loc, dir=dir, shape=shape, deg=deg)
            prompts.append(prompt)
            combinations.add(combo_key)
        
        if target_count > max_combos:
            print(f"提示：最大不重复组合数为{max_combos}，已返回全部组合")
        print(f"已生成{len(prompts)}个提示词（无放回抽样）")
    
    return prompts

def generate_prompts(target_count):
    base_prompt = "WELDSCALLOPS，在焊缝灰度图，位于焊道{pos}{loc}（收弧处）的位置添加一个凹坑缺陷： \ 
                   凹坑{dir}且轮廓呈{shape}，凹坑区域凹下去的梯度{deg}，\
                   凹坑仅出现在焊道上，灰度与背景适配。 "

    positions = ["上半部分", "下半部分"]
    locations = ["末端", "靠近末端"]
    directions = ["竖直", "水平", "倾斜"]
    shapes = ["扁圆形", "椭圆形", "不规则形状", "缝隙状"]
    degrees = [
        "明显（边缘到中心灰度骤降）且无反光（呈深黑色）",
        "平缓（边缘到中心灰度渐变）且有反光（灰色）"
    ]
    # type_weights=[0.3, 0.6, 0.1],  # 缺陷类型概率
    # shape_weights=[0.5, 0.3, 0.1, 0.1]  # 形态概率
    
    prompts = []
    combinations = set()
    max_combos = len(positions) * len(locations) * len(directions) * len(shapes) * len(degrees)
    actual_target = min(target_count, max_combos)
    
    while len(prompts) < actual_target:
        # 带权重随机选择参数
        # defect = random.choices(defect_types, weights=type_weights, k=1)[0]
        pos = random.choice(positions)
        loc = random.choice(locations)
        dir = random.choice(directions)
        shape = random.choice(shapes)
        deg = random.choice(degrees)
        
        # 去重键（避免完全重复的组合）
        combo_key = f"{pos}_{loc}_{dir}_{shape}_{deg}"
        if combo_key in combinations:
            continue
        
        # 生成最终prompt
        prompt = base_prompt.format(pos=pos, loc=loc, dir=dir, shape=shape, deg=deg)
        prompts.append(prompt)
        combinations.add(combo_key)
    
    if target_count > max_combos:
        print(f"提示：最大不重复组合数为{max_combos}，已返回全部组合")
    return prompts

def get_crop_coordinates(original_width, original_height, x1, y1, x2, y2, pipe):
    """
    处理用户输入的裁剪坐标，确保裁剪区域尺寸为16的倍数
    返回微调后的坐标和裁剪区域尺寸
    """
    # 确保坐标合法性（左上角<=右下角，且在原图范围内）
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0, min(x1, original_width))
    x2 = max(0, min(x2, original_width))
    y1 = max(0, min(y1, original_height))
    y2 = max(0, min(y2, original_height))
    
    # 计算原始裁剪尺寸
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # 使用模型方法调整尺寸为16的倍数
    adjusted_height, adjusted_width = pipe.check_resize_height_width(crop_height, crop_width)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    new_x1 = max(0, center_x - adjusted_width // 2)
    new_y1 = max(0, center_y - adjusted_height // 2)
    new_x2 = min(original_width, new_x1 + adjusted_width)
    new_y2 = min(original_height, new_y1 + adjusted_height)
    
    # 最终校验（避免因原图边界导致的尺寸偏差）
    # final_width = new_x2 - new_x1
    # final_height = new_y2 - new_y1
    # final_height, final_width = pipe.check_resize_height_width(final_height, final_width)

    # 如果因为边界限制导致尺寸变化，再次调整
    if (new_x2 - new_x1) != adjusted_width or (new_y2 - new_y1) != adjusted_height:
        # 重新计算实际尺寸并调整
        actual_width = new_x2 - new_x1
        actual_height = new_y2 - new_y1
        adjusted_height, adjusted_width = pipe.check_resize_height_width(actual_height, actual_width)
        # 再次调整坐标
        new_x1 = max(0, center_x - adjusted_width // 2)
        new_y1 = max(0, center_y - adjusted_height // 2)
        new_x2 = min(original_width, new_x1 + adjusted_width)
        new_y2 = min(original_height, new_y1 + adjusted_height)
    
    return (x1, y1, x2, y2), (new_x1, new_y1, new_x2, new_y2),  (adjusted_width, adjusted_height)

def crop_image(image, coordinates):
    x1, y1, x2, y2 = coordinates
    return image.crop((x1, y1, x2, y2))

def paste_cropped_image(original_image, cropped_image, coordinates):
    """将裁剪区域图片粘贴回原图"""
    x1, y1, x2, y2 = coordinates
    expected_width = x2 - x1
    expected_height = y2 - y1

    # 检查尺寸是否匹配
    if cropped_image.size != (expected_width, expected_height):
        print(f"警告: 裁剪图像尺寸{cropped_image.size}与坐标区域尺寸({expected_width}, {expected_height})不匹配")
        print(f"正在调整裁剪图像尺寸...")
        cropped_image = cropped_image.resize((expected_width, expected_height))
    # 确保尺寸匹配
    # cropped_resized = cropped_image.resize((x2 - x1, y2 - y1))
    
    result = original_image.copy()
    result.paste(cropped_image, (x1, y1))
    return result

def process_single_image(
    image_path,
    output_dir,
    lora_path,
    crop_coords,  # (x1, y1, x2, y2)
    num_inference_steps=10,
    seed=None
    ):
    # 创建输出目录结构
    hr_crop_dir = os.path.join(output_dir, "HR_Crop")
    hr_crop_enhanced_dir = os.path.join(output_dir, "HR_Crop_Enhanced")
    weld_enhanced_dir = os.path.join(output_dir, "WeldDefect_Enhanced")
    os.makedirs(hr_crop_dir, exist_ok=True)
    os.makedirs(hr_crop_enhanced_dir, exist_ok=True)
    os.makedirs(weld_enhanced_dir, exist_ok=True)
    
    # 加载模型管道
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    pipe.load_lora(pipe.dit, lora_path)
    
    # 加载原图
    try:
        original_image = Image.open(image_path).convert('RGB')
        original_width, original_height = original_image.size
        print(f"原图尺寸：{original_width}x{original_height}")
        print(f"用户输入裁剪坐标：{crop_coords}")
    except Exception as e:
        print(f"加载原图失败：{str(e)}")
        return
    
    # 处理裁剪坐标
    user_coords, adjusted_coords, (final_width, final_height) = get_crop_coordinates(
        original_width, original_height,
        crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3],
        pipe
    )
    print(f"微调后裁剪坐标：{adjusted_coords}")
    print(f"局部图尺寸：{final_width}x{final_height}")
    
    # 裁剪局部图并保存
    cropped_image = crop_image(original_image, adjusted_coords)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    crop_save_path = os.path.join(hr_crop_dir, f"{base_name}_cropped.jpg")
    cropped_image.save(crop_save_path)
    print(f"已保存裁剪局部图：{crop_save_path}")
    
    # 局部图增强推理
    try:
        # 调整尺寸并包装为列表（匹配模型输入格式）
        resized_crop = cropped_image.resize((final_width, final_height))
        input_images = [resized_crop]
        
        # 生成增强结果（可根据需求修改prompt）
        prompt = "生成焊接缺陷，保持原有场景结构，增强缺陷特征"
        with torch.no_grad():
            enhanced_crop = pipe(
                prompt,
                edit_image=input_images,
                seed=seed if seed is not None else random.randint(0, 10000),
                num_inference_steps=num_inference_steps,
                height=final_height,
                width=final_width
            )
        
        # 保存增强后的局部图
        enhanced_crop_save_path = os.path.join(hr_crop_enhanced_dir, f"{base_name}_cropped_enhanced.jpg")
        enhanced_crop.save(enhanced_crop_save_path)
        print(f"已保存增强局部图：{enhanced_crop_save_path}")
    except Exception as e:
        print(f"局部图增强失败：{str(e)}")
        return
    
    # 缝合回原图并保存
    try:
        final_image = paste_cropped_image(original_image, enhanced_crop, adjusted_coords)
        final_save_path = os.path.join(weld_enhanced_dir, f"{base_name}_enhanced.jpg")
        final_image.save(final_save_path)
        print(f"已保存最终增强图：{final_save_path}")
    except Exception as e:
        print(f"缝合图片失败：{str(e)}")
        return
    
    torch.cuda.empty_cache()

def load_crop_coords(json_path):
    """加载JSON坐标文件，返回{文件名: 坐标}字典"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"坐标文件不存在：{json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_dir(
    input_dir,
    output_dir,
    lora_path,
    coords_json,
    prompts_count=5,
    num_inference_steps=20,
    seed=None
    ):
    """
    批量处理目录下的图片：读取坐标→裁切→生成prompt→推理
    :param coords_json: 裁切坐标JSON文件路径
    """
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    hr_crop_dir = os.path.join(output_dir, "HR_Crop")
    hr_crop_enhanced_dir = os.path.join(output_dir, "HR_Crop_Enhanced")
    weld_enhanced_dir = os.path.join(output_dir, "WeldDefect_Enhanced")
    os.makedirs(hr_crop_dir, exist_ok=True)
    os.makedirs(hr_crop_enhanced_dir, exist_ok=True)
    os.makedirs(weld_enhanced_dir, exist_ok=True)
    
    coords_dict = load_crop_coords(coords_json)
    
    # 加载模型管道（复用原有逻辑）
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    pipe.load_lora(pipe.dit, lora_path)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    total_images = len([f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)])
    
    for img_idx, img_filename in enumerate(os.listdir(input_dir)):
        if not img_filename.lower().endswith(image_extensions):
            continue
        
        img_path = os.path.join(input_dir, img_filename)
        try:
            print(f"\n===== 处理第 {img_idx+1}/{total_images} 张：{img_filename} =====")
            original_image = Image.open(img_path).convert('RGB')
            original_width, original_height = original_image.size
            print(f"原图尺寸：{original_width}x{original_height}")
            
            # 获取裁切坐标（优先用图片专属坐标，否则用default）
            coords = coords_dict.get(img_filename, coords_dict.get("default"))
            print(f"json裁剪坐标：{coords}")
            if not coords:
                raise ValueError(f"未找到{img_filename}的裁切坐标，且无default配置")
            # 裁剪坐标需要微调以适应qwen-image-edit-2509
            # 处理裁剪坐标
            user_coords, adjusted_coords, (final_width, final_height) = get_crop_coordinates(
                original_width, original_height,
                coords.get('x1'), coords.get('y1'), coords.get('x2'), coords.get('y2'),
                pipe
            )
            print(f"微调后裁剪坐标：{adjusted_coords}")
            print(f"局部图尺寸：{final_width}x{final_height}")

            # 裁剪局部图并保存
            cropped_image = crop_image(original_image, adjusted_coords)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            crop_save_path = os.path.join(hr_crop_dir, f"{base_name}_cropped.jpg")
            cropped_image.save(crop_save_path)
            print(f"已保存裁剪局部图：{crop_save_path}")
            
            # 生成该图片的专属prompt列表
            prompts = generate_prompts(target_count=prompts_count)
            print(f"生成{len(prompts)}个prompt：{prompts}...") 

            # 调整尺寸并包装为列表（匹配模型输入格式）
            resized_crop = cropped_image.resize((final_width, final_height))
            input_images = [resized_crop]
            
            # 遍历prompt推理并保存
            for prompt_idx, prompt in enumerate(prompts):
                with torch.no_grad():
                    enhanced_crop = pipe(
                        prompt,
                        edit_image=input_images, 
                        seed=seed,
                        num_inference_steps=num_inference_steps,
                        height=final_height,
                        width=final_width
                    )
                
                # 保存增强后的局部图
                enhanced_crop_save_path = os.path.join(hr_crop_enhanced_dir, f"{base_name}_cropped_prompt{prompt_idx}.jpg")
                enhanced_crop.save(enhanced_crop_save_path)
                print(f"已保存增强局部图：{enhanced_crop_save_path}")

                # 缝合回原图并保存
                final_image = paste_cropped_image(original_image, enhanced_crop, adjusted_coords)
                final_save_path = os.path.join(weld_enhanced_dir, f"{base_name}_enhanced_prompt{prompt_idx}.jpg")
                final_image.save(final_save_path)
                print(f"已保存最终增强图：{final_save_path}")
                
                
        except Exception as e:
            print(f"处理{img_filename}失败：{str(e)}")
            torch.cuda.empty_cache()
            continue
    
    torch.cuda.empty_cache()
    print("\n批量处理完成！")

def main():
    parser = argparse.ArgumentParser(description='焊接缺陷局部增强工具（裁剪-增强-缝合流程）')
    parser.add_argument('image_path', help='输入图片路径')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('lora_path', help='LoRA模型路径')
    parser.add_argument('--x1', type=int, required=True, help='裁剪区域左上角x坐标')
    parser.add_argument('--y1', type=int, required=True, help='裁剪区域左上角y坐标')
    parser.add_argument('--x2', type=int, required=True, help='裁剪区域右下角x坐标')
    parser.add_argument('--y2', type=int, required=True, help='裁剪区域右下角y坐标')
    parser.add_argument('--steps', type=int, default=10, help='推理步数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（默认随机）')
    
    args = parser.parse_args()
    
    # 执行处理
    process_single_image(
        image_path=args.image_path,
        output_dir=args.output_dir,
        lora_path=args.lora_path,
        crop_coords=(args.x1, args.y1, args.x2, args.y2),
        num_inference_steps=args.steps,
        seed=args.seed
    )

def batch_main():
    parser = argparse.ArgumentParser(description="焊缝缺陷局部增强工具")
    parser.add_argument("--input-dir", type=str, default="path/to/input", help="输入图片目录")
    parser.add_argument("--output-dir", type=str, default="path/to/output", help="输出结果目录")
    parser.add_argument("--lora-path", type=str, default="path/to/lora", help="lora路径")
    parser.add_argument("--coords-json", type=str, default="path/to/coords", help="坐标配置JSON文件路径")
    parser.add_argument("--prompts-count", type=int, default=3, help="生成的prompt数量")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="推理步数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.isdir(args.input):
        print(f"错误：输入目录 {args.input} 不存在")
        return
    
    if not os.path.isfile(args.coords):
        print(f"错误：坐标文件 {args.coords} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 执行批量处理
    process_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lora_path=args.lora_path,
        coords_json=args.coords_json,
        prompts_count=args.prompts_count,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed
    )

if __name__ == "__main__":
    batch_main()
