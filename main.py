import os

os.environ["KERAS_BACKEND"] = "torch"

import json

import numpy as np
import keras
import gradio as gr

MODEL_DIR = "models/CNN/MobileNetV3Small"

model = keras.models.load_model(
    f"{MODEL_DIR}/MobileNetV3Small_hand_gestures_detection.keras"
)

with open(f"{MODEL_DIR}/classes.json") as f:
    id2label: dict[str, str] = json.load(f)

GESTURE_EMOTIONS = {
    "call": "Общительный",
    "dislike": "Негативный",
    "fist": "Уверенный",
    "four": "Нейтральный",
    "like": "Позитивный",
    "mute": "Сдержанный",
    "ok": "Позитивный",
    "one": "Нейтральный",
    "palm": "Нейтральный",
    "peace": "Миролюбивый",
    "peace_inverted": "Враждебный",
    "rock": "Энергичный",
    "stop": "Тревожный",
    "stop_inverted": "Тревожный",
    "three": "Нейтральный",
    "three2": "Нейтральный",
    "two_up": "Миролюбивый",
    "two_up_inverted": "Враждебный",
}


def predict(image):
    if image is None:
        return {}

    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.efficientnet.preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]

    return {
        f"{GESTURE_EMOTIONS[id2label[str(i)]]} ({id2label[str(i)]})": float(p)
        for i, p in enumerate(preds)
    }


with gr.Blocks(title="Распознавание жестов рук") as demo:
    gr.Markdown("# Распознавание жестов рук\nЗагрузите фото или используйте камеру")

    with gr.Tab("Загрузить фото"):
        with gr.Row():
            upload_image = gr.Image(type="pil", sources=["upload"])
            upload_output = gr.Label(num_top_classes=5)

        upload_btn = gr.Button("Распознать")
        upload_btn.click(predict, inputs=upload_image, outputs=upload_output)

    with gr.Tab("Камера"):
        with gr.Row():
            cam_image = gr.Image(
                type="pil", sources=["webcam"], streaming=True, scale=1,
            )
            cam_output = gr.Label(num_top_classes=5, scale=1)
        cam_image.stream(predict, inputs=cam_image, outputs=cam_output)

if __name__ == "__main__":
    demo.launch()
