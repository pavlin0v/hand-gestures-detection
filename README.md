# Hand Gestures Recognition

Веб-приложение для распознавания жестов рук с помощью CNN (MobileNetV3Small). Классифицирует 18 типов жестов и определяет связанную эмоциональную окраску.

## Поддерживаемые жесты

`call` `dislike` `fist` `four` `like` `mute` `ok` `one` `palm` `peace` `peace_inverted` `rock` `stop` `stop_inverted` `three` `three2` `two_up` `two_up_inverted`

## Стек

- **Keras** (бэкенд PyTorch) — инференс модели
- **Gradio** — веб-интерфейс (загрузка фото / камера в реальном времени)

## Запуск

```bash
uv sync
uv run main.py
```

Приложение откроется на `http://localhost:7860`.
