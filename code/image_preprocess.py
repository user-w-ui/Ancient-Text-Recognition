import os
import cv2
import numpy as np
from pathlib import Path

def cv_imread(file_path):
    """使用 imdecode 读取图像，完美绕过 Windows 中文/复杂路径报错"""
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

def cv_imwrite(file_path, img):
    """使用 imencode 保存图像"""
    cv2.imencode('.jpg', img)[1].tofile(file_path)

def auto_crop_margins(image):
    """自动裁剪扫描图像的纯白边距"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image 
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # 留 15 像素的安全边距，别切太狠伤到武术小人
    padding = 15
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    return image[y1:y2, x1:x2]

def process_images():
    # 基础路径
    input_base_dir = Path(r"D:\DSML\Computer Vision RA\data\page_1st_batch")
    output_base_dir = Path(r"D:\DSML\Computer Vision RA\data\page_1st_batch-processed")
    
    # 排除的复杂书籍 
    exclude_book = "slgf_lres"
    
    # 使用 rglob 递归搜索所有子文件夹里的图片
    image_paths = list(input_base_dir.rglob("*.[jp][pn]g")) 
    
    if not image_paths:
        print("❌ 没找到图片！检查下路径是不是写错了？")
        return

    processed_count = 0
    
    for img_path in image_paths:
        # 检查当前图片的路径中是否包含要排除的书名 
        if exclude_book in img_path.parts:
            continue
            
        # 计算相对路径，为了在输出目录中重建一模一样的子文件夹结构
        rel_path = img_path.relative_to(input_base_dir)
        output_path = output_base_dir / rel_path
        
        # 提前把对应的子文件夹建好
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取图像
        img = cv_imread(str(img_path))
        if img is None:
            print(f"⚠️ 读取失败，跳过: {rel_path}")
            continue
            
        h, w = img.shape[:2]
        
        # 1. 旋转处理：遇到横版（如 bl_mg ）直接掰直
        if w > h:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # print(f"🔄 旋转了: {rel_path.name}")
            
        # 2. 智能裁剪大白边：特别是 wbz_mg 这种边距离谱的 
        img = auto_crop_margins(img)
        
        # 3. 统一缩放到高度 800px
        new_h = 800
        old_h, old_w = img.shape[:2]
        new_w = int((new_h / old_h) * old_w)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 4. 保存
        cv_imwrite(str(output_path), img_resized)
        processed_count += 1
        
        # 每处理 50 张报个平安，免得你以为死机了
        if processed_count % 50 == 0:
            print(f"⏳ 正在肝... 已处理 {processed_count} 张图")
            
    print(f"\n✅ 终于搞定！完美处理了 {processed_count} 张图片。")
    print(f"📂 快去这看看成果: {output_base_dir}")

if __name__ == "__main__":
    process_images()