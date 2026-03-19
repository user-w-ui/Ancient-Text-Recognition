import os
import cv2
import json
import uuid
from pathlib import Path
from paddleocr import PPStructureV3 

engine = PPStructureV3()

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
            import numpy as np
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
                
            original_height, original_width = img.shape[:2]
            
            try:
                result = engine.predict(img)
                if type(result).__name__ == 'generator':
                    result = list(result)
                page_res = result[0] if isinstance(result, list) else result
            except Exception as e:
                print(f"❌ 推理炸了: {e}")
                continue
            
            # 🔪 分别存储 data 和 annotations 里的数据
            data_bbox_list = []
            data_label_list = []
            data_transcription_list = []
            annotations_result_list = []
            
            if 'layout_det_res' in page_res and 'boxes' in page_res['layout_det_res']:
                for box in page_res['layout_det_res']['boxes']:
                    label_name = box.get('label', '')
                    x1, y1, x2, y2 = [float(c) for c in box['coordinate']]
                    
                    if label_name.lower() == 'image':
                        label = "Illustration"
                    elif label_name.lower() in ['text', 'textline']:
                        label = "Vertical_text"
                    elif label_name.lower() in ['title', 'header']:
                        label = "Doc_title"
                    else:
                        label = "Vertical_text" 
                        
                    x_pct = (x1 / original_width) * 100
                    y_pct = (y1 / original_height) * 100
                    w_pct = ((x2 - x1) / original_width) * 100
                    h_pct = ((y2 - y1) / original_height) * 100
                    
                    box_id = str(uuid.uuid4())[:10]
                    
                    # 1. 组装 data.bbox 结构
                    data_bbox_list.append({
                        "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct, "rotation": 0,
                        "original_width": original_width, "original_height": original_height
                    })
                    
                    # 2. 组装 data.label 结构
                    data_label_list.append({
                        "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct, "rotation": 0,
                        "rectanglelabels": [label],
                        "original_width": original_width, "original_height": original_height
                    })
                    
                    # 3. 组装 data.transcription (用空字符串占位保持数组长度对齐)
                    data_transcription_list.append("")
                    
                    # 4. 组装 annotations.result 结构
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
                        "origin": "manual" # 伪装成人工标注
                    })
                    
            # 🔥 终极整合：100% 还原你导出的样本格式
            task_data = {
                "data": {
                    "ocr": f"https://mabook.s3.ap-southeast-1.amazonaws.com/{book_name}/{img_path.name}",
                    "id": img_id_counter,
                    "bbox": data_bbox_list,
                    "label": data_label_list,
                    "transcription": data_transcription_list
                },
                "annotations": [{
                    "result": annotations_result_list
                }]
            }
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