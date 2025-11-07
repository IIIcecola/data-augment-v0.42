from gradio_client import Client, handle_file
import os
import argparse
from tqdm import tqdm
import traceback

# -------------------------- 核心配置（需根据实际情况修改）--------------------------
API_URL = "your_actual_pusa_ti2v_api_url" 
# 视频生成Prompt列表
VIDEO_PROMPT_LIST = [
    "摄像机视角稳定，延续首帧背景，工人双手抓握攀爬物，双脚交替平稳踩踏向上攀爬，动作连贯无跳跃，正常光线，人物清晰，无近景特写，符合工业摄像机拍摄质感"
]
# 支持的图片格式（可根据需要扩展）
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# -------------------------- 工具函数 --------------------------
def get_all_image_files(input_path):
    """获取输入路径下所有支持的图片文件（包括子目录）"""
    image_files = []
    if os.path.isfile(input_path):
        # 单张图片
        if input_path.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            image_files.append(input_path)
        else:
            print(f"警告：{input_path} 不是支持的图片格式，跳过")
    elif os.path.isdir(input_path):
        # 目录（遍历所有子目录）
        print(f"正在扫描目录 {input_path} 及其子目录下的图片...")
        for ext in SUPPORTED_IMAGE_FORMATS:
            # 使用rglob遍历所有子目录的图片
            for img_path in input_path.rglob(f"*{ext}"):
                image_files.append(str(img_path))
        print(f"共发现 {len(image_files)} 张图片")
    else:
        raise ValueError(f"错误：输入路径 {input_path} 不是文件或目录")
    return image_files

def generate_video(client, img_path, video_prompt, output_dir):
    """调用API生成单个视频并保存"""
    try:
        # 调用图生视频API（沿用已测试的参数）
        result = client.predict(
            prompt=video_prompt,
            negative_prompt='',
            seed=1,
            steps=4,
            input_image=handle_file(img_path),
            end_image=None,
            mode_selector="图生视频",  # 修正原代码的中文逗号问题
            fps_slider=24,
            input_video=None,
            prompt_refiner=False,
            lora_selector=[],
            height=720,
            width=1280,
            frame_num=81,
            api_name="/generate_video"
        )

        # 解析API返回的视频路径
        video_temp_path = result.get("video")
        if not video_temp_path or not os.path.exists(video_temp_path):
            print(f"警告：API未返回有效视频路径 {img_path}")
            return False

        # 构建输出视频路径
        img_dir, img_name = os.path.split(img_path)
        img_base_name, img_ext = os.path.splitext(img_name)
        output_video_name = f"{img_base_name}.mp4"
        output_video_path = os.path.join(output_dir, output_video_name)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 复制临时视频到输出目录（避免临时文件被清理）
        with open(video_temp_path, "rb") as f_in, open(output_video_path, "wb") as f_out:
            f_out.write(f_in.read())

        print(f"成功生成视频：{output_video_path}")
        return True

    except Exception as e:
        print(f"\n错误：生成视频失败 {img_path}")
        print(f"错误详情：{str(e)}")
        traceback.print_exc()
        return False

# -------------------------- 主流程 --------------------------
def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='批量图生视频工具（Pusa TI2V API）- 异常攀高项目第二步骤')
    parser.add_argument('input', help='输入路径（单张图片路径或图片目录，支持子目录遍历）')
    parser.add_argument('output', help='输出视频目录（自动创建）')
    parser.add_argument('--api-url', default=API_URL, help=f'Pusa TI2V API地址（默认：{API_URL}）')
    parser.add_argument('--cycle-prompt', action='store_true', help='当图片数量超过prompt数量时，循环使用prompt列表')

    args = parser.parse_args()

    # 初始化API客户端（全局初始化，避免重复创建连接）
    print(f"连接API：{args.api_url}")
    try:
        client = Client(args.api_url, timeout=300)  # 超时设置为5分钟（适应视频生成耗时）
    except Exception as e:
        print(f"错误：无法连接API {args.api_url}")
        traceback.print_exc()
        return

    # 获取所有待处理图片
    try:
        image_files = get_all_image_files(args.input)
        if not image_files:
            print("错误：未找到任何支持的图片文件")
            return
    except Exception as e:
        print(f"错误：获取图片文件失败")
        traceback.print_exc()
        return

    # 批量生成视频（带进度条）
    print(f"\n开始批量生成视频（共 {len(image_files)} 张图片）...")
    success_count = 0
    fail_count = 0

    for img_path in tqdm(image_files, desc="视频生成进度"):
        for prompt in VIDEO_PROMPT_LIST:
          try:
            generate_video(client, img_path, video_prompt, args.output)
          except Exception as e:
            print(f"error: {str(e)}")
            traceback.print_exc()

    # 输出统计结果
    print("\n" + "="*50)
    print(f"批量处理完成！")
    print(f"总处理图片：{len(image_files)} 张")
    print(f"输出目录：{os.path.abspath(args.output)}")
    print("="*50)

if __name__ == "__main__":
    main()
