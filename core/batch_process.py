import os
import random
import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

def crack_prompts(target_count):
    """
    生成SMT电路板裂缝缺陷的随机Prompt列表（控制程度概率）
    :param target_count: 需要生成的Prompt数量
    :return: 去重后的随机Prompt列表
    """
    # 基础指令前缀（固定）
    base_instruction = "SMTCRACK, 在SMT电路板的{位置}区域生成{程度}缺陷。"
    
    # 可选位置
    positions = [
        "左上角", "左下角", "右上角", "右下角",
        "竖直中央", "水平中央", "上边", "下边", "左边", "右边"
    ]
    
    # ========== 核心修改：带概率的程度选择 ==========
    # 程度列表 + 对应概率权重（裂缝30%、局部裂缝60%、大范围裂缝10%）
    degrees_list = ["裂缝", "局部裂缝", "大范围裂缝"]
    degrees_weights = [0.3, 0.6, 0.1]  # 权重和建议为1，也可写[3,6,1]（比例一致即可）
    
    # 生成组合并去重
    prompts = []
    combinations = set()
    max_combinations = len(positions) * len(degrees_list)  # 10×3=30
    actual_target = min(target_count, max_combinations)
    
    while len(prompts) < actual_target:
        # 随机选位置（等概率）
        pos = random.choice(positions)
        # 带权重选程度（30%/60%/10%）
        deg = random.choices(degrees_list, weights=degrees_weights, k=1)[0]
        
        # 去重逻辑不变
        combo_key = f"{pos}_{deg}"
        if combo_key in combinations:
            continue
        
        prompt = base_instruction.format(位置=pos, 程度=deg)
        combinations.add(combo_key)
        prompts.append(prompt)
    
    if target_count > max_combinations:
        print(f"提示：最大可生成不重复Prompt数为 {max_combinations}，已返回全部不重复组合")
    
    return prompts

def get_valid_dimensions(original_width, original_height, use_original_size, custom_width, custom_height, pipe):
    """获取符合模型要求的尺寸（确保是16的倍数）"""
    if use_original_size:
        target_width, target_height = original_width, original_height
    else:
        target_width, target_height = custom_width, custom_height
    return pipe.check_resize_height_width(target_height, target_width)

def batch_process(
    input_dir, 
    output_dir, 
    lora_path,
    target_prompt_count,
    num_inference_steps=10, 
    seed=None,
    use_original_size=True,
    custom_width=1024,
    custom_height=1024,
    ):
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载管道
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    
    pipe.load_lora(pipe.dit, lora_path)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    for img_idx, img_filename in enumerate(os.listdir(input_dir)):
        if not img_filename.lower().endswith(image_extensions):
            continue
        
        img_path = os.path.join(input_dir, img_filename)
        try:
            image = Image.open(img_path).convert('RGB')
            original_width, original_height = image.size
            print(f"\n===== 处理第 {img_idx+1} 张图：{img_filename}（原始尺寸：{original_width}x{original_height}） =====")
            
            # 计算有效尺寸
            output_height, output_width = get_valid_dimensions(
                original_width, original_height, use_original_size, custom_width, custom_height, pipe
            )
            print(f"调整后输出尺寸：{output_width}x{output_height}")
            
            # 每张图生成独立的带概率Prompt列表
            current_prompts = crack_prompts(target_prompt_count)
            print(f"为该图生成 {len(current_prompts)} 个随机Prompt：{current_prompts}")
            
            # 调整图片尺寸并包装为列表（关键：匹配官方输入格式）
            resized_image = image.resize((output_width, output_height))
            images = [resized_image]  # 列表包装，触发LoRA生效
            
            # 遍历Prompt生成变体
            for prompt_idx, prompt in enumerate(current_prompts):
                with torch.no_grad():
                    edited_image = pipe(
                        prompt,
                        edit_image=images,  # 传入列表格式的图片
                        seed=seed,
                        num_inference_steps=num_inference_steps,
                        height=output_height,
                        width=output_width
                    )
                
                # 保存结果
                base_name = os.path.splitext(img_filename)[0]
                save_path = os.path.join(output_dir, f"{base_name}_img{img_idx}_prompt{prompt_idx}.jpg")
                edited_image.save(save_path)
                print(f"已保存：{save_path}（尺寸：{edited_image.size}）")
                
        except Exception as e:
            print(f"处理 {img_filename} 时出错：{str(e)}")
            torch.cuda.empty_cache()
            continue
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = "path/to/your/smt/images"
    OUTPUT_DIR = "path/to/your/crack_variants"
    TARGET_PROMPT_COUNT = 3  # 每张图生成8个Prompt
    INFERENCE_STEPS = 10
    SEED = None
    USE_ORIGINAL_SIZE = True
    CUSTOM_WIDTH = 1024
    CUSTOM_HEIGHT = 1024
    
    # LoRA配置
    LORA_PATH = "models/train/Qwen-Image-Edit-2509_lora/epoch-4.safetensors"
    
    # 执行批量处理
    batch_process(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        lora_path=LORA_PATH,
        target_prompt_count=TARGET_PROMPT_COUNT,
        num_inference_steps=INFERENCE_STEPS,
        seed=SEED,
        use_original_size=USE_ORIGINAL_SIZE,
        custom_width=CUSTOM_WIDTH,
        custom_height=CUSTOM_HEIGHT
    )
