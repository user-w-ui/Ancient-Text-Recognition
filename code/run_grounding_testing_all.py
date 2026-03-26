from pathlib import Path

from grounding import process_books


INPUT_ROOT = Path(r"D:\DSML\Computer Vision RA\data\testing set")
OUTPUT_ROOT_BINRZ = Path(r"D:\DSML\Computer Vision RA\data\testing_set_bbox_binrz")
OUTPUT_ROOT_RAW = Path(r"D:\DSML\Computer Vision RA\data\testing_set_bbox")

# 需要处理的 testing set 一级目录
SPLITS = ["handBY", "handCR", "printBW", "printBY"]

# 如需排除某个书籍目录，可在这里配置，例如: {"handBY": "slgf_lres"}
EXCLUDE_BOOK_BY_SPLIT = {}


if __name__ == "__main__":
    use_binrz = input("是否在预测前做二值化？(y/n): ").strip().lower() in {"y", "yes", "1", "true"}
    output_root = OUTPUT_ROOT_BINRZ if use_binrz else OUTPUT_ROOT_RAW
    print(f"已选择二值化: {use_binrz}")
    print("已默认开启整图级判断(img_judge)")
    print(f"结果将保存到: {output_root}")

    for split in SPLITS:
        input_dir = INPUT_ROOT / split
        output_dir = output_root / split
        exclude_book = EXCLUDE_BOOK_BY_SPLIT.get(split)

        print(f"\n========== 开始处理 {split} ==========")
        process_books(
            input_base_dir=input_dir,
            output_base_dir=output_dir,
            exclude_book=exclude_book,
            if_binrz=use_binrz,
            img_judge_default="待审核",
        )
        print(f"========== 完成 {split} ==========")
