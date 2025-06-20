import serial
import time
import sys

def control_arduino(port, command):
    """
    指定されたシリアルポートを通じてArduinoにコマンドを送信する関数
    """
    try:
        # シリアルポートを開く
        # ボーレートはArduino側の Serial.begin() の設定と合わせる (例: 9600)
        ser = serial.Serial(port, 9600, timeout=2)
        # Arduinoが起動し、シリアル通信が安定するまで少し待つ
        time.sleep(2)

        # コマンドをバイト形式で送信
        ser.write(command.encode('utf-8'))
        print(f"Arduino@{port} へコマンド '{command}' を送信しました。")

    except serial.SerialException as e:
        # エラー処理
        print(f"エラー: シリアルポート '{port}' を開けませんでした。")
        print(e)
        # 呼び出し元にエラーを伝えるために終了コード1で終了
        sys.exit(1)

    finally:
        # ポートが正常に開かれていれば、閉じる
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("シリアルポートを閉じました。")

if __name__ == '__main__':
    # このスクリプトが直接実行されたときの処理
    # コマンドラインから引数を2つ受け取ることを想定 (例: python arduino_control.py COM3 M)
    if len(sys.argv) < 3:
        print("使い方: python arduino_control.py <COMポート名> <送信コマンド>")
        # 引数が足りない場合は終了
        sys.exit(1)

    com_port = sys.argv[1]    # 1番目の引数をCOMポート名として使用
    command_to_send = sys.argv[2] # 2番目の引数を送信コマンドとして使用

    # Arduino制御関数を呼び出し
    control_arduino(com_port, command_to_send)