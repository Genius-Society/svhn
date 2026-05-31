import os
import torch
import random
import warnings
import gradio as gr
from PIL import Image
from model import Model
from torchvision import transforms

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"

if EN_US:
    import huggingface_hub

    MODEL_DIR = huggingface_hub.snapshot_download(
        "Genius-Society/svhn",
        cache_dir="./__pycache__",
    )

else:
    import modelscope

    MODEL_DIR = modelscope.snapshot_download(
        "Genius-Society/svhn",
        cache_dir="./__pycache__",
    )
ZH2EN = {
    "上传图片": "Upload an image",
    "状态栏": "Status",
    "选择模型": "Select a model",
    "识别结果": "Recognition result",
    "门牌号识别": "Door Number Recognition",
}


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


def infer(input_img: str, checkpoint_file: str):
    status = "Success"
    outstr = ""
    try:
        model = Model()
        model.restore(f"{MODEL_DIR}/{checkpoint_file}")
        with torch.no_grad():
            transform = transforms.Compose(
                [
                    transforms.Resize([64, 64]),
                    transforms.CenterCrop([54, 54]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            image = Image.open(input_img)
            image = image.convert("RGB")
            image = transform(image)
            images = image.unsqueeze(dim=0)
            (
                length_logits,
                digit1_logits,
                digit2_logits,
                digit3_logits,
                digit4_logits,
                digit5_logits,
            ) = model.eval()(images)
            length_prediction = length_logits.max(1)[1]
            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]
            digit3_prediction = digit3_logits.max(1)[1]
            digit4_prediction = digit4_logits.max(1)[1]
            digit5_prediction = digit5_logits.max(1)[1]
            output = [
                digit1_prediction.item(),
                digit2_prediction.item(),
                digit3_prediction.item(),
                digit4_prediction.item(),
                digit5_prediction.item(),
            ]

            for i in range(length_prediction.item()):
                outstr += str(output[i])

    except Exception as e:
        status = f"{e}"

    return status, outstr


def get_files(dir_path=MODEL_DIR, ext=".pth"):
    files_and_folders = os.listdir(dir_path)
    outputs = []
    for file in files_and_folders:
        if file.endswith(ext):
            outputs.append(file)

    return outputs


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    models = get_files()
    images = get_files(f"{MODEL_DIR}/examples", ".png")
    samples = []
    for img in images:
        samples.append(
            [
                f"{MODEL_DIR}/examples/{img}",
                models[random.randint(0, len(models) - 1)],
            ]
        )

    gr.Interface(
        fn=infer,
        inputs=[
            gr.Image(label=_L("上传图片"), type="filepath"),
            gr.Dropdown(
                label=_L("选择模型"),
                choices=models,
                value=models[0],
            ),
        ],
        outputs=[
            gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
            gr.Textbox(label=_L("识别结果"), buttons=["copy"]),
        ],
        examples=samples,
        title=_L("门牌号识别"),
        flagging_mode="never",
        cache_examples=False,
    ).launch(
        theme=gr.themes.Monochrome(),
        css="#gradio-share-link-button-0 { display: none; }",
        ssr_mode=False,
    )
