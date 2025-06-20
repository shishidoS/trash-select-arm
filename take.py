import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import subprocess

# GPUが利用可能か確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# YOLOv10モデルのロード（GPUを使用）
model = YOLO("yolov10x.pt").to(device)

# 分類対象（ターゲット）
target_pra = ["bottle","cup"]  # プラごみ
target_nama = ["orange","book"]  # 燃えるゴミ
target_reso = ["cell phone"]  # 資源ごみ

# 日本語フォントのロード
font_path = "C:/Windows/Fonts/msgothic.ttc"  # MSゴシック
try:
    font = ImageFont.truetype(font_path, 40)  # 40pxのフォントサイズ
except IOError:
    print("フォントが見つかりません。デフォルトフォントを使用します。")
    font = ImageFont.load_default()

# USBカメラのキャプチャ（インデックス0）
cap = cv2.VideoCapture(0)

# カメラの解像度を設定（例: 1280x720）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラからフレームを取得できませんでした")
        break

    # YOLOで物体認識（GPUで推論）
    results = model(frame)  
    result = results[0]

    # OpenCV画像をPIL画像に変換
    annotated_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_pil)

    for box in result.boxes:
        cls = int(box.cls.item())  # クラスID取得
        label = model.names[cls]  # クラス名取得
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 座標を取得

        if label in target_pra:
            display_label = "プラごみ:" + label
            color = (255, 0, 0)  # 赤
        elif label in target_nama:
            display_label = "燃えるゴミ:" + label
            color = (255, 165, 0)  # オレンジ
            try:
                command_to_send = 'M' # 燃えるゴミ(Moeru)を意味するコマンド
                print(f"'{label}'を検知。Arduino制御プログラムを起動します...")
                
                # subprocess.runで外部のPythonスクリプトを実行
                # ['python', スクリプト名, 引数1, 引数2] のようにリストで指定
                subprocess.run(
                    ['python', ARDUINO_CONTROL_SCRIPT, ARDUINO_PORT, command_to_send],
                    check=True, # 実行が失敗した場合にエラーを発生させる
                    capture_output=True, # 実行結果の出力をキャプチャする
                    text=True # 出力をテキストとして扱う
                )
                print("Arduinoの動作が完了しました。")

            except FileNotFoundError:
                print(f"エラー: 制御スクリプト '{ARDUINO_CONTROL_SCRIPT}' が見つかりません。")
            except subprocess.CalledProcessError as e:
                print(f"エラー: Arduino制御スクリプトの実行に失敗しました。")
                print(f"--- 制御スクリプトからの出力 ---\n{e.stdout}\n{e.stderr}") 
            
        elif label in target_reso:
            display_label = "資源ごみ:" + label
            color = (0, 255, 0)  # 緑
        else:
            display_label = label
            color = (0, 0, 255)  # 青

        # 枠を描画
        thickness = 4 if label in target_pra + target_nama else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color[::-1], thickness)

        # 日本語テキストを描画
        draw.text((x1, y1 - 40), display_label, font=font, fill=color)

    # PIL画像をOpenCV画像に戻す
    annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

    # 結果を表示
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
