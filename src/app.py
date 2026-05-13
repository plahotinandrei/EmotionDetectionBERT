import gradio as gr
from inference import predict_emotion

def classify(text):
    emotion, confidence = predict_emotion(text)
    return emotion, confidence

iface = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Введите текст на русском языке...",
        label="Текст для анализа"
    ),
    outputs=[
        gr.Label(label="Распознанная эмоция"),
        gr.Number(label="Уверенность")
    ],
    title="Распознавание эмоций в тексте",
    description="""
    Введите короткий текст на русском языке, и модель определит,
    какая эмоция в нём выражена (радость, грусть, удивление, страх, гнев или нейтрально).
    Модель основана на дообученном `DeepPavlov/rubert-base-cased`.
    """,
    examples=[
        ["Я так счастлив, что не могу уснуть!"],
        ["Мне очень грустно расставаться с вами."],
        ["Как ты это сделал? Я потрясён!"],
        ["Я боюсь завтрашнего экзамена."],
        ["Меня бесит, когда опаздывают!"],
        ["Сегодня обычный день, ничего особенного."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()