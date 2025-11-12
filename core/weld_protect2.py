from gradio_client import Client, handle_file
import shutil
import os
import argparse
from tqdm import tqdm
import random

# 初始化API客户端（根据实际地址调整）
client = Client("http://10.59.67.2:5012/")

def find_image_files(root_dir):
    """查找目录下所有图片文件（优先甲方提供的45张完整监控图）"""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                image_files.append(os.path.join(dirpath, filename))
    # 过滤掉可能的非监控图（可选：根据甲方文件名规则筛选，比如包含"监控"关键词）
    image_files = [f for f in image_files if "监控" in os.path.basename(f) or "完整" in os.path.basename(f)]
    return image_files

def process_monitor_images(source, output_dir, target_width=1920, target_height=1080, adjust_light=True):
    """
    基于完整监控原图，仅修改人物属性+强化未佩戴防护
    :param source: 甲方45张完整监控图的目录/单张图片
    :param output_dir: 输出目录
    :param target_width/height: 保持原图分辨率（默认1920*1080）
    :param adjust_light: 是否调整环境光线（默认True，可通过参数关闭）
    """
    # 仅保留需要修改的核心属性组合（避免改动场景）
    clothes = [
        "蓝色工装服", "红色安全背心+深色长裤", "黄色反光马甲+灰色工装裤",
        "灰色长袖工作服", "黑色耐磨夹克+卡其裤", "绿色劳保服",
        "深蓝色连体工装", "橙色安全服+工作靴", "白色衬衫+深蓝色工作裤",
        "迷彩工作服", "棕色工装裤+蓝色长袖上衣"
    ]
    
    body_types = [
        "体型中等的", "体型偏瘦的", "体型偏胖的",
        "身材高大的", "身材矮小的", "体型健壮的"
    ]
    
    ages = [
        "20-30岁的年轻人", "30-40岁的中年人",
        "40-50岁的中年人", "50-60岁的老年人"
    ]
    
    genders = ["男性", "女性"]
    
    # 可选光线调整（如需保持原图光线，可设为仅1项或通过参数关闭）
    light_conditions = [
        "保持原图光线不变",
        "正常室内工业光线（均匀明亮）",
        "弱光环境（可清晰辨识人物，无过度黑暗）"
    ] if adjust_light else ["保持原图光线不变"]

    # 确定处理对象（甲方45张完整图）
    if os.path.isdir(source):
        image_files = find_image_files(source)
        print(f"发现 {len(image_files)} 张完整监控图片，开始批量处理...")
        if len(image_files) == 0:
            print("错误：未找到有效监控图片（请检查目录或文件名是否包含'监控'关键词）")
            return
    elif os.path.isfile(source) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        image_files = [source]
        print(f"开始处理单张完整监控图片: {source}")
    else:
        print(f"错误: 无效的图片源 - {source}")
        return

    # 批量处理：每张原图生成多组人物属性组合
    for image_path in tqdm(image_files, desc="处理进度"):
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        
        # 随机生成N组属性组合（每张原图生成5-10组，避免过度冗余）
        num_variations = 400  # 可调整：每张原图生成的变体数量
        for var_idx in range(num_variations):
            # 随机选择属性（保证多样性）
            cloth = random.choice(clothes)
            body = random.choice(body_types)
            age = random.choice(ages)
            gender = random.choice(genders)
            light = random.choice(light_conditions)
            
            # 核心Prompt：严格限制「仅修改人物，不改动场景」
            prompt = (
                f"严格保留原图的完整监控视角、背景环境、设备布局、画面比例和监控质感，不做任何改动。"
                f"仅替换原图中的焊接人员：将其修改为{age}{gender}，{body}，穿着{cloth}。"
                f"光线调整：{light}。"
                f"核心要求：替换后的人物需保持与原图人物相同的作业姿势和位置，"
                f"面部清晰可见，明显未佩戴任何面部防护装备（无护目镜、无面罩、无口罩），"
                f"人物比例与原图一致，融入场景自然，无违和感，焊接动作和火花效果保留原图特征。"
            )
            
            # 生成唯一ID用于命名
            param_id = hash(f"{cloth}{body}{age}{gender}{light}{var_idx}") % 100000
            edit_one_person(
                client,
                image_path,
                prompt,
                param_id,
                var_idx,
                output_dir,
                target_width,
                target_height
            )

def edit_one_person(client, image_path, prompt, param_id, var_idx, output_path, target_width, target_height):
    """仅替换图片中的人物属性，保留其他所有元素"""
    try:
        # 调用API进行人物替换（优化参数保证Prompt执行）
        result = client.predict(
            image1=handle_file(image_path),  # 传入完整监控原图
            image2=None,
            image3=None,
            prompt=prompt,
            seed=random.randint(1, 1000000),
            randomize_seed=True,
            true_guidance_scale=1.2,  # 提高引导度，确保仅修改人物
            num_inference_steps=5,  # 增加推理步数，提升人物融合度
            rewrite_prompt=False,
            height=target_height,
            width=target_width,
            api_name="/infer"
        )
        src_path = result[0]

        # 保存结果（命名包含变体信息，方便追溯）
        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        dst_path = os.path.join(output_path, f"{name}_var{var_idx}_pid{param_id}{ext}")
        shutil.move(src_path, dst_path)
        return True
    except Exception as e:
        print(f"\n处理 {image_path} 变体{var_idx} 失败：{str(e)}")
    return False

def main():
    parser = argparse.ArgumentParser(description='焊接防护缺失数据增强工具（基于完整监控图）')
    parser.add_argument('source', help='甲方完整监控图片源（单张图片路径或目录）')
    parser.add_argument('output', help='处理结果输出目录')
    parser.add_argument('--width', type=int, default=1920, help='输出图片宽度（默认1920，建议保持原图）')
    parser.add_argument('--height', type=int, default=1080, help='输出图片高度（默认1080，建议保持原图）')
    parser.add_argument('--no-light', action='store_true', help='不调整光线（保持原图光线）')
    
    args = parser.parse_args()

    process_monitor_images(
        source=args.source,
        output_dir=args.output,
        target_width=args.width,
        target_height=args.height,
        adjust_light=not args.no_light  # 控制是否调整光线
    )

if __name__ == "__main__":
    main()
