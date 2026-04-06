import torch
import gradio as gr
from transformers import ViTForImageClassification, ViTImageProcessor

model = ViTForImageClassification.from_pretrained("./my_model")
processor = ViTImageProcessor.from_pretrained("./my_model")
model.eval()

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

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    return {
        f"{GESTURE_EMOTIONS[model.config.id2label[i]]} ({model.config.id2label[i]})": prob.item()
        for i, prob in enumerate(probs)
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
