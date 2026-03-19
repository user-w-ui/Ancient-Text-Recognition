import cv2
import json
import uuid
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
from paddleocr import PPStructureV3 

engine = PPStructureV3()


def _map_layout_label(label_name: str) -> str:
    normalized = label_name.lower()
    if normalized == 'image':
        return "Illustration"
    if normalized in ['text', 'textline']:
        return "Vertical_text"
    if normalized in ['title', 'header']:
        return "Doc_title"
    return "Vertical_text"


def build_task_from_image(img_path: Path, book_name: str, task_id: int) -> Optional[Dict[str, Any]]:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    original_height, original_width = img.shape[:2]

    try:
        result = engine.predict(img)
        if type(result).__name__ == 'generator':
            result = list(result)
        page_res = result[0] if isinstance(result, list) else result
    except Exception as e:
        print(f"❌ 推理炸了: {e}")
        return None

    annotations_result_list = []
    if 'layout_det_res' in page_res and 'boxes' in page_res['layout_det_res']:
        for box in page_res['layout_det_res']['boxes']:
            x1, y1, x2, y2 = [float(c) for c in box['coordinate']]
            label = _map_layout_label(box.get('label', ''))

            x_pct = (x1 / original_width) * 100
            y_pct = (y1 / original_height) * 100
            w_pct = ((x2 - x1) / original_width) * 100
            h_pct = ((y2 - y1) / original_height) * 100

            box_id = str(uuid.uuid4())[:10]
            annotations_result_list.append({
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": [label]
                },
                "id": box_id,
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "manual"
            })

    return {
        "id": task_id,
        "data": {
            "ocr": f"https://mabook.s3.ap-southeast-1.amazonaws.com/{book_name}/{img_path.name}"
        },
        "annotations": [{
            "result": annotations_result_list
        }]
    }

def process_books():
    input_base_dir = Path(r"D:\DSML\Computer Vision RA\data\page_1st_batch-processed")
    output_base_dir = Path(r"D:\DSML\Computer Vision RA\data\page_1st_batch_bbox")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    exclude_book = "slgf_lres"
    book_folders = [d for d in input_base_dir.iterdir() if d.is_dir()]
    
    for book_folder in book_folders:
        book_name = book_folder.name
        if book_name == exclude_book:
            continue
            
        if book_name != "bdj_qm":
            continue
        
        print(f"🚀 开始处理书籍: {book_name} ...")
        image_paths = list(book_folder.glob("*.[jp][pn]g"))
        
        book_tasks_for_ls = []
        img_id_counter = 1
        
        for img_path in image_paths:
            task_data = build_task_from_image(img_path, book_name, img_id_counter)
            if task_data is None:
                continue

            book_tasks_for_ls.append(task_data)
            
            if img_id_counter % 10 == 0:
                print(f"⏳ {book_name} 进度: 已打框 {img_id_counter} 张图")
            img_id_counter += 1
                
        if book_tasks_for_ls:
            book_json_path = output_base_dir / f"{book_name}_bbox.json"
            with open(book_json_path, 'w', encoding='utf-8') as f:
                json.dump(book_tasks_for_ls, f, ensure_ascii=False, indent=2)
            print(f"✅ {book_name} 搞定！数据躺在: {book_json_path}\n")

if __name__ == "__main__":
    process_books()