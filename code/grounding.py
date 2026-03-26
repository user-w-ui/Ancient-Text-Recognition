import cv2
import json
import uuid
import argparse
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
from paddleocr import PPStructureV3 

# 得把旋转检测和文档矫正关掉，否则输出的预测框会错位，因为后续要基于原图的坐标来计算百分比坐标
engine = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)


def _map_layout_label(label_name: str) -> str:
    normalized = label_name.lower()
    if normalized == 'image':
        return "Illustration"
    if normalized in ['text', 'textline']:
        return "Vertical_text"
    if normalized in ['title', 'header']:
        return "Doc_title"
    return "Vertical_text"


def _binarize_for_layout(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 模型通常要求3通道输入；这里复制通道，不会改变二值化后的前景/背景信息
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _build_image_level_choice_result(
    original_width: int,
    original_height: int,
    default_choice: str,
) -> Dict[str, Any]:
    # 这个结果不依附任何框，对应Label Studio里perRegion="false"的Choices控件
    return {
        "original_width": original_width,
        "original_height": original_height,
        "image_rotation": 0,
        "value": {
            "choices": [default_choice]
        },
        "id": str(uuid.uuid4())[:10],
        "from_name": "img_judge",
        "to_name": "image",
        "type": "choices",
        "origin": "prediction"
    }


def build_task_from_image(
    img_path: Path,
    book_name: str,
    task_id: int,
    if_binrz: bool = False,
    img_judge_default: str = "待审核",
) -> Optional[Dict[str, Any]]:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    # 记录原图尺寸，后续计算百分比坐标需要
    original_height, original_width = img.shape[:2]
    # 如果需要二值化，则在推理前先处理一下图像
    model_input = _binarize_for_layout(img) if if_binrz else img

    try:
        result = engine.predict(model_input)
        if type(result).__name__ == 'generator':
            result = list(result)
        page_res = result[0] if isinstance(result, list) else result
    except Exception as e:
        print(f"❌ 推理炸了: {e}")
        return None

    # 组装JSON以符合Label Studio的格式，坐标转换为百分比，并且映射标签名称
    prediction_result_list = []
    if 'layout_det_res' in page_res and 'boxes' in page_res['layout_det_res']:
        for box in page_res['layout_det_res']['boxes']:
            x1, y1, x2, y2 = [float(c) for c in box['coordinate']]
            label = _map_layout_label(box.get('label', ''))

            x_pct = (x1 / original_width) * 100
            y_pct = (y1 / original_height) * 100
            w_pct = ((x2 - x1) / original_width) * 100
            h_pct = ((y2 - y1) / original_height) * 100

            box_id = str(uuid.uuid4())[:10]
            prediction_result_list.append({
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
                "origin": "prediction"
            })

    prediction_result_list.append(
        _build_image_level_choice_result(
            original_width=original_width,
            original_height=original_height,
            default_choice=img_judge_default,
        )
    )

    return {
        "id": task_id,
        "data": {
            "ocr": f"https://mabook.s3.ap-southeast-1.amazonaws.com/{book_name}/{img_path.name}"
        },
        "predictions": [{
            "model_version": "PPStructureV3",
            "result": prediction_result_list
        }]
    }

def process_books(
    input_base_dir: Path,
    output_base_dir: Path,
    exclude_book: Optional[str] = None,
    if_binrz: bool = False,
    img_judge_default: str = "待审核",
) -> None:
    if not input_base_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_base_dir}")

    output_base_dir.mkdir(parents=True, exist_ok=True)

    book_folders = [d for d in input_base_dir.iterdir() if d.is_dir()]
    
    for book_folder in book_folders:
        book_name = book_folder.name
        if book_name == exclude_book:
            continue
            
        
        print(f"🚀 开始处理书籍: {book_name} ...")
        image_paths = list(book_folder.glob("*.[jp][pn]g"))
        
        book_tasks_for_ls = []
        img_id_counter = 1
        
        for img_path in image_paths:
            task_data = build_task_from_image(
                img_path,
                book_name,
                img_id_counter,
                if_binrz=if_binrz,
                img_judge_default=img_judge_default,
            )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layout grounding and export Label Studio JSON.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Base directory containing book folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save *_bbox.json files.")
    parser.add_argument("--exclude-book", type=str, default=None, help="Optional book folder name to skip.")
    parser.add_argument("--if-binrz", action="store_true", help="Apply Otsu binarization before prediction.")
    parser.add_argument("--img-judge-default", type=str, default="待审核", help="Default choice value for img_judge.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_books(
        input_base_dir=args.input_dir,
        output_base_dir=args.output_dir,
        exclude_book=args.exclude_book,
        if_binrz=args.if_binrz,
        img_judge_default=args.img_judge_default,
    )

if __name__ == "__main__":
    main()