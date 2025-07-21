import os
import logging
from typing import Dict, List, Tuple

import cv2
import easyocr
import fitz
import numpy as np
import pandas as pd
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

PDF_PATH: str = "C:/amu/AmuStage/learn/data/Bessarab.pdf"
TOLERANCE: int = 5
STR_TOLERANCE: float = 0.2
AREA_TOLERANCE: float = 0.10
HORIZONTAL_KERNEL_SIZE = (40, 1)
VERTICAL_KERNEL_SIZE = (1, 40)
HOUGH_THRESHOLD = 100
MIN_LINE_LENGTH = 5
MAX_LINE_GAP = 5
MERGE_THRESHOLD = 5
reader = easyocr.Reader(['ru', 'en'])
LOG_FILE = "extraction.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


def contains(outer: tuple[int], inner: tuple[int]) -> bool:
    x0, y0, w0, h0 = outer
    x1, y1, w1, h1 = inner
    return (x0 <= x1 and y0 <= y1 and
            x0 + w0 >= x1 + w1 and
            y0 + h0 >= y1 + h1)


def merge_close_lines(lines, is_horizontal=True, merge_threshold=10):
    if lines is None or len(lines) == 0:
        return lines
    lines_sorted = sorted(
        lines, key=lambda x: x[0][1] if is_horizontal else x[0][0]
    )
    merged_lines = []
    current_group = [lines_sorted[0][0]]
    for line in lines_sorted[1:]:
        line = line[0]
        ref_current = (current_group[-1][1] if is_horizontal
                       else current_group[-1][0])
        ref_line = line[1] if is_horizontal else line[0]
        if abs(ref_line - ref_current) <= merge_threshold:
            current_group.append(line)
        else:
            if is_horizontal:
                y_median = np.median([_[1] for _ in current_group])
                merged_line = [
                    current_group[0][0], int(y_median),
                    current_group[0][2], int(y_median)
                ]
            else:
                x_median = np.median([_[0] for _ in current_group])
                merged_line = [
                    int(x_median), current_group[0][1],
                    int(x_median), current_group[0][3]
                ]
            merged_lines.append([merged_line])
            current_group = [line]
    if is_horizontal:
        y_median = np.median([_[1] for _ in current_group])
        merged_line = [
            current_group[0][0], int(y_median),
            current_group[0][2], int(y_median)
        ]
    else:
        x_median = np.median([_[0] for _ in current_group])
        merged_line = [
            int(x_median), current_group[0][1],
            int(x_median), current_group[0][3]
        ]
    merged_lines.append([merged_line])
    return merged_lines


def correct_ocr_errors(text):
    replacements = {
        'O': '0', 'o': '0',
        'l': '1', 'I': '1',
        'Z': '2', 'z': '2',
        'A': '4', 'q': '9',
        ' ': ''
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text


def is_digit_cell(text):
    return any(c.isdigit() for c in text)


def extract_text_from_cell(page, bbox, text_dict):
    cell_text = []
    for block in text_dict['blocks']:
        if 'lines' in block:
            for line in block['lines']:
                for span in line['spans']:
                    span_bbox = span['bbox']
                    if (bbox[0] <= span_bbox[0] and bbox[1] <= span_bbox[1] and
                            bbox[2] >= span_bbox[2]
                            and bbox[3] >= span_bbox[3]):
                        cell_text.append(span['text'].strip())
    standard_result = ' '.join(cell_text).strip()
    if not standard_result or is_digit_cell(standard_result):
        pix = page.get_pixmap(clip=bbox, dpi=600)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_processed = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]
        img_processed = cv2.erode(
            img_processed, np.ones((2, 2), np.uint8), iterations=1
        )
        ocr_result = reader.readtext(img_processed, detail=0)
        ocr_text = correct_ocr_errors(' '.join(ocr_result).strip())
        if ocr_text:
            return ocr_text
    return standard_result


def group_text_into_rows(boxes, tolerance=5):
    rows = []
    current_row = []
    current_y = None
    for box in sorted(boxes, key=lambda b: b[1]):
        if current_y is None or abs(box[1] - current_y) < tolerance:
            current_row.append(box)
            current_y = box[1] if current_y is None else (
                current_y + box[1]) / 2
        else:
            rows.append(current_row)
            current_row = [box]
            current_y = box[1]
    if current_row:
        rows.append(current_row)
    return rows


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    all_tables = []
    doc: fitz.Document = fitz.open(PDF_PATH)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_dict: Dict = page.get_text("dict")
        pix = page.get_pixmap()
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )
        image = cv2.bilateralFilter(image, 10, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=50, tileGridSize=(50, 50))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, thresh_value = cv2.threshold(
            gray_image, 50, 255, cv2.THRESH_BINARY_INV
        )
        kernel = np.ones((2, 2), np.uint8)
        dilated_value = cv2.dilate(thresh_value, kernel, iterations=6)
        dilated_value = cv2.GaussianBlur(dilated_value, (3, 3), 0)
        contours, _ = cv2.findContours(
            dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1
        )
        rects = [cv2.boundingRect(contour) for contour in contours]
        height, width = image.shape[:2]
        page_area = width * height
        min_area = page_area * AREA_TOLERANCE
        outer_rects = []
        for i, rect0 in enumerate(rects):
            x, y, w, h = rect0
            area = w * h
            contains_other = any(
                i != j and contains(rect0, rect1)
                for j, rect1 in enumerate(rects)
            )
            not_contained = all(
                i == j or not contains(rect1, rect0)
                for j, rect1 in enumerate(rects)
            )
            if contains_other and not_contained and area >= min_area:
                outer_rects.append(rect0)
        table_vertical_lines = {}
        for table_idx, (x, y, w, h) in enumerate(outer_rects):
            roi = gray_image[y:y + h, x:x + w]
            img_blur = cv2.GaussianBlur(roi, (3, 3), 0)
            _, thresh = cv2.threshold(
                img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            kernel_vert = cv2.getStructuringElement(
                cv2.MORPH_RECT, VERTICAL_KERNEL_SIZE
            )
            vertical = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel_vert, iterations=1
            )
            lines_vert = cv2.HoughLinesP(
                vertical, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
                minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP
            )
            if lines_vert is not None:
                lines_vert = merge_close_lines(
                    lines_vert, is_horizontal=False,
                    merge_threshold=MERGE_THRESHOLD
                )
            table_vertical_lines[table_idx] = {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'vertical': lines_vert
            }
        boxes: List[Tuple[float, float, float, float]] = [
            span['bbox']
            for block in text_dict['blocks'] if 'lines' in block
            for line in block['lines']
            for span in line['spans']
        ]
        for table_idx, table_data in table_vertical_lines.items():
            x_table = table_data['x']
            y_table = table_data['y']
            w_table = table_data['w']
            h_table = table_data['h']
            lines_vert = table_data['vertical']
            if lines_vert is None:
                continue
            columns = [0]
            for line in lines_vert:
                x1, _, _, _ = line[0]
                columns.append(x1)
            columns.append(w_table)
            columns = sorted(list(set(columns)))
            table_boxes = [
                b for b in boxes
                if x_table <= b[0] and y_table <= b[1]
                and (x_table + w_table) >= b[2]
                and (y_table + h_table) >= b[3]
            ]
            text_rows = group_text_into_rows(table_boxes, TOLERANCE)
            rows = []
            for row in text_rows:
                if not row:
                    continue
                min_y = min(b[1] for b in row)
                rows.append(min_y - y_table)
            rows.append(h_table)
            rows = sorted(rows)
            rows_with_coords = []
            for i in range(len(rows) - 1):
                row_cells = []
                for j in range(len(columns) - 1):
                    cell_x0 = x_table + columns[j]
                    cell_y0 = y_table + rows[i]
                    cell_x1 = x_table + columns[j + 1]
                    cell_y1 = y_table + rows[i + 1]
                    cell_text = extract_text_from_cell(
                        page,
                        (cell_x0, cell_y0, cell_x1, cell_y1),
                        text_dict
                    )
                    row_cells.append({
                        'x0': cell_x0,
                        'y0': cell_y0,
                        'x1': cell_x1,
                        'y1': cell_y1,
                        'text': cell_text
                    })
                rows_with_coords.append(row_cells)
            data_row_idx = None
            for i, row in enumerate(rows_with_coords):
                if any(any(char.isdigit() for char in cell['text']) for cell in row):
                    data_row_idx = i
                    break
            if data_row_idx is not None and data_row_idx > 0:
                parent_dict = {}
                for i in range(len(rows_with_coords) - 1, 0, -1):
                    for child_idx, child in enumerate(rows_with_coords[i]):
                        best_parent = None
                        best_overlap = 0
                        for parent_idx, parent in enumerate(rows_with_coords[i - 1]):
                            overlap = max(
                                0, min(child['x1'], parent['x1']) - max(child['x0'], parent['x0'])
                            )
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_parent = (i - 1, parent_idx)
                        if best_parent and best_overlap > 0:
                            parent_dict[(i, child_idx)] = best_parent

                def get_full_header(cell_row, cell_col):
                    headers = []
                    current_row = cell_row
                    current_col = cell_col
                    while current_row >= 0:
                        current_cell = rows_with_coords[current_row][current_col]
                        headers.append(current_cell['text'])
                        parent_found = False
                        for (row, col), parent in parent_dict.items():
                            if row == current_row and col == current_col:
                                current_row, current_col = parent
                                parent_found = True
                                break
                        if not parent_found:
                            break
                    return " | ".join(reversed(headers)) if headers else ""

                headers_for_columns = []
                for j in range(len(columns) - 1):
                    header = get_full_header(data_row_idx, j)
                    headers_for_columns.append(
                        header if header else rows_with_coords[0][j]['text']
                    )
                data = []
                for row in rows_with_coords[data_row_idx + 1:]:
                    data.append([cell['text'] for cell in row])
                df = pd.DataFrame(data, columns=headers_for_columns)
            else:
                headers = [cell['text'] for cell in rows_with_coords[0]] if len(
                    rows_with_coords) > 1 else None
                data = []
                for row in rows_with_coords[1:]:
                    data.append([cell['text'] for cell in row])
                df = pd.DataFrame(data, columns=headers)
            logging.info(
                f"Таблица {table_idx + 1} на странице {page_num + 1} извлечена"
            )
            all_tables.append({
                'page': page_num + 1,
                'table_num': table_idx + 1,
                'table': df
            })
    doc.close()
    if all_tables:
        output_dir = os.path.join(os.getcwd(), "extracted_tables_csv")
        os.makedirs(output_dir, exist_ok=True)
        for table_info in all_tables:
            filename = (
                f"extracted_table_page_{table_info['page']}_"
                f"table_{table_info['table_num']}.csv"
            )
            table_info['table'].to_csv(
                os.path.join(output_dir, filename), index=False
            )
        logging.info(f"Все таблицы сохранены в директорию {output_dir}")
    else:
        logging.warning("Не найдено таблиц для сохранения")


if __name__ == "__main__":
    main()
