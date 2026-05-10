import cv2 as cv
import numpy as np 
from pyzbar.pyzbar import decode
from qreader import QReader
from zxingcpp import read_barcodes
import time 
import os 


def create_test_frames(file_name, saving_folder): # Функция для разбиения видео на кадры 
    vid = cv.VideoCapture(file_name)

    count, success = 0, True
    while success:
        success, image = vid.read()

        if success:

            cv.imwrite(f"{saving_folder}test_frame_{count}.jpg", image)
            count += 1
    vid.release()

def test_pyzbar(folder_path, saving_folder): # Тест библиотеки pyzbar
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    image_count = len(image_files)

    start_time = time.time()
    detections_per_frame = []
    unique_codes_all = set()
    frame_to_codes = {} 
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        img = cv.imread(image_path)

        pyzbar_results = decode(img)

        codes_on_frame = []
        for result in pyzbar_results:
            text = result.data.decode('utf-8', errors='ignore')
            rect = result.rect
            codes_on_frame.append(text)
            unique_codes_all.add(text)

            cv.rectangle(img, (rect.left, rect.top), 
                        (rect.left + rect.width, rect.top + rect.height), (255, 0, 0), 10)

        if pyzbar_results:
            cv.imwrite(f"{saving_folder}/annotated_{filename}", img)
        
        detections_per_frame.append(len(pyzbar_results))
        frame_to_codes[idx] = codes_on_frame
    
    end_time = time.time()

    total_time = end_time - start_time
    avg_fps = image_count / total_time

    avg_detections = sum(detections_per_frame) / image_count
    frames_with_codes = sum(1 for x in detections_per_frame if x > 0)

    unique_count = len(unique_codes_all)
    print(f"  Всего кадров: {image_count}")
    print(f"  Общее время: {total_time:.2f} сек")
    print(f"  Средняя скорость: {avg_fps:.2f} FPS")
    print(f"  Среднее время на кадр: {total_time/image_count*1000:.2f} мс")
    print(f"  Кадров с кодами: {frames_with_codes} из {image_count} ({frames_with_codes/image_count*100:.1f}%)")
    print(f"  Среднее кодов на кадр: {avg_detections:.2f}")
    print(f"  Всего найденных дублей: {sum(detections_per_frame)}")
    print(f"  Уникальных кодов: {unique_count}")
    print(f"Уникальные коды: {unique_codes_all}")

def test_zxing(folder_path, saving_folder):  # Тест библиотеки zxing
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    image_count = len(image_files)

    start_time = time.time()
    detections_per_frame = []
    unique_codes_all = set()
    unique_codes_with_info = []
    frame_to_codes = {} 
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        img = cv.imread(image_path)

        zxing_results = read_barcodes(img)

        codes_on_frame = []
        for result in zxing_results:
            text = result.text
            barcode_format = str(result.format)
            position = result.position
            
            codes_on_frame.append(text)
            unique_codes_all.add(text)

            pts = [
                [position.top_left.x, position.top_left.y],
                [position.top_right.x, position.top_right.y],
                [position.bottom_right.x, position.bottom_right.y],
                [position.bottom_left.x, position.bottom_left.y]
            ]
            pts = np.array(pts, dtype=np.int32)
            cv.polylines(img, [pts], True, (255, 0, 0), 10)

        if zxing_results:
            cv.imwrite(f"{saving_folder}/annotated_{filename}", img)
        
        detections_per_frame.append(len(zxing_results))
        frame_to_codes[idx] = codes_on_frame
    
    end_time = time.time()

    total_time = end_time - start_time
    avg_fps = image_count / total_time

    avg_detections = sum(detections_per_frame) / image_count
    frames_with_codes = sum(1 for x in detections_per_frame if x > 0)

    unique_count = len(unique_codes_all)

    print(f"  Всего кадров: {image_count}")
    print(f"  Общее время: {total_time:.2f} сек")
    print(f"  Средняя скорость: {avg_fps:.2f} FPS")
    print(f"  Среднее время на кадр: {total_time/image_count*1000:.2f} мс")

    print(f"  Кадров с кодами: {frames_with_codes} из {image_count} ({frames_with_codes/image_count*100:.1f}%)")
    print(f"  Среднее кодов на кадр: {avg_detections:.2f}")
    print(f"  Всего обнаружений (с дублями): {sum(detections_per_frame)}")
    print(f"  Уникальных кодов: {unique_count}")


def test_qreader(folder_path, saving_folder):  # Тест библиотеки qreader
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    image_count = len(image_files)

    qreader = QReader(model_size="m")
    
    start_time = time.time()
    detections_per_frame = []
    unique_codes_all = set()
    unique_codes_with_info = []
    frame_to_codes = {}
    codes_with_no_text = 0
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        img = cv.imread(image_path)

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        decoded_texts, decoded_info = qreader.detect_and_decode(img_rgb, return_detections=True)
        
        codes_on_frame = []
        
        for j in range(len(decoded_texts)):
            text, bbox, confidence = decoded_texts[j], decoded_info[j]["bbox_xyxy"], decoded_info[j]["confidence"]
            if text is None: 
                codes_with_no_text += 1
                continue
            
            codes_on_frame.append(text)
            unique_codes_all.add(text)
            unique_codes_with_info.append({'text': text, 'confidence': confidence})
            if len(bbox) == 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                if confidence > 0.8:
                    color = (0, 255, 0)
                elif confidence > 0.5:
                    color = (0, 255, 255)  
                else:
                    color = (0, 0, 255)  
                cv.rectangle(img, (x1, y1), (x2, y2), color, 10)
                cv.putText(img, f"{confidence:.2f}", (x1, y1 - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        if codes_on_frame: 
            os.makedirs(saving_folder, exist_ok=True)
            cv.imwrite(f"{saving_folder}/annotated_{filename}", img)
        
        detections_per_frame.append(len(codes_on_frame))
        frame_to_codes[idx] = codes_on_frame
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_fps = image_count / total_time
    
    avg_detections = sum(detections_per_frame) / image_count if image_count > 0 else 0
    frames_with_codes = sum(1 for x in detections_per_frame if x > 0)
    unique_count = len(unique_codes_all)

    confidences_list = [info['confidence'] for info in unique_codes_with_info]
    avg_confidence = sum(confidences_list) / len(confidences_list) if confidences_list else 0
    
    print(f"  Всего кадров: {image_count}")
    print(f"  Общее время: {total_time:.2f} сек")
    print(f"  Средняя скорость: {avg_fps:.2f} FPS")
    print(f"  Среднее время на кадр: {total_time/image_count*1000:.2f} мс")
    print(f"  Кадров с кодами: {frames_with_codes} из {image_count} ({frames_with_codes/image_count*100:.1f}%)")
    print(f"  Среднее кодов на кадр: {avg_detections:.2f}")
    print(f"  Всего обнаружений (с дублями): {sum(detections_per_frame)}")
    print(f"  Уникальных кодов: {unique_count}")
    print(f"  Средняя уверенность: {avg_confidence:.3f}")
    print(f"Число кодов с нераспозанным текстом {codes_with_no_text}")

