import os
import uuid
import asyncio
import logging
from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import edge_tts
import requests
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import uvicorn

# --- 配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Generator")

# 挂载静态目录，用于访问生成的视频
# 请确保运行目录下有一个 'static' 文件夹，如果没有代码会自动创建
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- 数据模型 ---
class SceneItem(BaseModel):
    image_url: str
    text: str


class VideoRequest(BaseModel):
    scenes: List[SceneItem]
    voice: str = "zh-CN-XiaoxiaoNeural"  # 默认 Edge-TTS 语音


# --- 辅助函数：下载图片 ---
def download_image(url: str, save_path: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"下载图片失败 {url}: {e}")
        return False


# --- 新增辅助函数：自动换行算法 ---
def text_wrap(text, font, max_width, draw):
    """
    根据指定宽度自动换行
    """
    lines = []

    # 如果文本包含换行符，先按换行符物理切分
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue

        current_line = ""
        for char in paragraph:
            # 试探性加上这个字符，算算宽度
            test_line = current_line + char
            # 获取包围盒 (left, top, right, bottom)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_w = bbox[2] - bbox[0]

            if text_w <= max_width:
                # 没超，继续加
                current_line = test_line
            else:
                # 超了，把当前行存入，新起一行
                lines.append(current_line)
                current_line = char
        # 把最后剩下的也存入
        if current_line:
            lines.append(current_line)

    return lines


# --- 修改后的图片处理函数 ---
def process_image_with_text(image_path: str, text: str, output_path: str):
    """
    将图片调整为 1080x1920，并绘制自动换行的字幕
    """
    target_w, target_h = 1080, 1920
    # 设置文本最大宽度 (例如屏幕宽度的 85%)，留出左右边距
    max_text_width = int(target_w * 0.85)

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # --- 1. 图片缩放裁切逻辑 (保持不变) ---
            img_ratio = img.width / img.height
            target_ratio = target_w / target_h

            if img_ratio > target_ratio:
                new_h = target_h
                new_w = int(img.width * (target_h / img.height))
            else:
                new_w = target_w
                new_h = int(img.height * (target_w / img.width))

            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_w) / 2
            top = (new_h - target_h) / 2
            img = img.crop((left, top, left + target_w, top + target_h))

            # --- 2. 字幕绘制逻辑 (核心修改) ---
            draw = ImageDraw.Draw(img, "RGBA")

            # 字体加载 (尝试加载 font.ttf，否则默认)
            font_size = 60
            try:
                # 建议行高为字体大小的 1.2 倍
                line_height = int(font_size * 1.5)
                if os.path.exists("font.ttf"):
                    font = ImageFont.truetype("font.ttf", font_size)
                else:
                    # Windows 系统尝试找黑体
                    font = ImageFont.truetype("simhei.ttf", font_size)
            except:
                font = ImageFont.load_default()
                line_height = font_size + 10

            # --- 关键步骤：计算自动换行 ---
            lines = text_wrap(text, font, max_text_width, draw)

            # 计算整个文本块的总高度
            total_text_height = len(lines) * line_height

            # 字幕背景框的高度 (多留点上下 padding)
            padding_v = 40
            bg_h = total_text_height + (padding_v * 2)

            # 背景框的 Y 坐标 (屏幕底部向上偏移 200像素)
            bg_y1 = target_h - bg_h - 200
            bg_y2 = target_h - 200

            # 绘制半透明黑色背景
            # 左右留一点边距，或者全宽
            bg_x1 = (target_w - max_text_width) / 2 - 20  # 背景比文字宽一点
            bg_x2 = target_w - bg_x1

            draw.rectangle(
                [(bg_x1, bg_y1), (bg_x2, bg_y2)],
                fill=(0, 0, 0, 160),  # 黑色，透明度 160/255
                outline=None,
                width=0,
                # 圆角 (Pillow 新版支持 corners, 旧版可能不支持，这里先画矩形)
            )

            # --- 逐行绘制文字 ---
            current_y = bg_y1 + padding_v

            for line in lines:
                # 计算每一行的宽度，以便居中
                bbox = draw.textbbox((0, 0), line, font=font)
                line_w = bbox[2] - bbox[0]

                # 居中坐标
                line_x = (target_w - line_w) / 2

                # 绘制描边 (让文字更清晰)
                stroke_width = 2
                stroke_fill = "black"

                draw.text((line_x, current_y), line, font=font, fill="white",
                          stroke_width=stroke_width, stroke_fill=stroke_fill)

                current_y += line_height

            img.save(output_path)
            return True
    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        return False

# --- 核心逻辑：生成视频 ---
async def generate_video_task(request: VideoRequest, task_id: str):
    task_dir = os.path.join(STATIC_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    clips = []

    try:
        for idx, scene in enumerate(request.scenes):
            # 1. 准备文件路径
            raw_img_path = os.path.join(task_dir, f"raw_{idx}.jpg")
            processed_img_path = os.path.join(task_dir, f"proc_{idx}.jpg")
            audio_path = os.path.join(task_dir, f"audio_{idx}.mp3")

            # 2. 下载图片
            if not download_image(scene.image_url, raw_img_path):
                continue

            # 3. 处理图片（画字幕，转尺寸）
            # 为了更好的效果，建议下载一个支持中文的字体重命名为 font.ttf 放在项目根目录
            process_image_with_text(raw_img_path, scene.text, processed_img_path)

            # 4. 生成语音 (Edge-TTS)
            communicate = edge_tts.Communicate(scene.text, request.voice)
            await communicate.save(audio_path)

            # 5. 合成单片段 (Image + Audio)
            # 加载音频
            audio_clip = AudioFileClip(audio_path)
            # 加载图片，设置时长等于音频时长
            img_clip = ImageClip(processed_img_path).set_duration(audio_clip.duration)
            img_clip = img_clip.set_audio(audio_clip)
            # 设置淡入淡出效果让过渡自然
            img_clip = img_clip.crossfadein(0.5)

            clips.append(img_clip)

        if not clips:
            raise Exception("没有有效的片段生成")

        # 6. 拼接所有片段
        final_clip = concatenate_videoclips(clips, method="compose")  # compose 处理转场

        # 7. 导出视频
        output_filename = f"{task_id}.mp4"
        output_path = os.path.join(STATIC_DIR, output_filename)

        # 使用线程池运行 write_videofile，防止阻塞 FastAPI 的事件循环
        # 注意：在 Windows 上 MoviePy 可能需要 codec='libx264'
        # fps=24 足够短视频使用
        await asyncio.to_thread(
            final_clip.write_videofile,
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            logger=None  # 关闭详细日志输出
        )

        # 清理内存中的 Clips
        final_clip.close()
        for c in clips:
            c.close()

        logger.info(f"视频生成成功: {output_path}")
        return output_filename

    except Exception as e:
        logger.error(f"视频生成任务失败: {e}")
        # 实际生产中这里应该更新数据库状态
        raise e


# --- 接口定义 ---

@app.post("/generate_video")
async def create_video(request: VideoRequest):
    """
    接收 JSON，同步生成视频（注意：长视频会阻塞 HTTP 请求，建议生产环境改为异步任务+轮询）
    为了演示简单，这里演示等待结果返回。
    """
    task_id = str(uuid.uuid4())
    logger.info(f"收到任务 {task_id}, 包含 {len(request.scenes)} 个场景")

    try:
        filename = await generate_video_task(request, task_id)

        # 构建完整的下载 URL
        # 假设部署在本地，host 为 localhost:8000
        download_url = f"http://localhost:8000/static/{filename}"

        return {
            "status": "success",
            "task_id": task_id,
            "download_url": download_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)