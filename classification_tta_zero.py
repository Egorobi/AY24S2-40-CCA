import unitouch.ImageBind.data as data
from unitouch.ImageBind.models.x2touch_model_part import x2touch,imagebind_huge
import torch
import os
import torchvision.transforms.functional as F
from torchvision.transforms import v2

llama_dir = "./unitouch/llama_ori"
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["CUDA_HOME"] = "/usr/lib/cuda"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/cuda/lib64"

def augment(touch, num_views=3):
    augmentation = v2.Compose([v2.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.5, 1.5)),
                               v2.RandomChoice(transforms=[
                                    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                                    v2.RandomAdjustSharpness(p=1, sharpness_factor=2)]),
                                # v2.RandomHorizontalFlip(p=0.3),
                                # v2.RandomVerticalFlip(p=0.3)
                               ])
    views = [touch]
    for n in range(num_views-1):
        copy = touch.detach().clone()
        views.append(augmentation(copy))
    return torch.stack(views, dim=0)

def confidence_filter(logits: torch.Tensor, probs: torch.Tensor, top:float, return_idx: bool=False):
    batch_entropy = -(probs * probs.log()).sum(1)
    full_idx = torch.argsort(batch_entropy, descending=False)
    filt_idx = full_idx[:int(batch_entropy.size()[0] * top)]
    if not return_idx:
        return logits[filt_idx]
    return logits[filt_idx], filt_idx, full_idx

def zero(model, touch, z_txt, N=64, gamma=0.3):
    # step1: augment
    views = augment(touch, num_views=N)
    # step2: predict (unscaled logits)
    inputs = {
    "touch": views
    }
    embeddings = model(inputs)
    logits = embeddings["touch"] @ z_txt.t()
    # print("TTA Logits: " + str(logits))
    # step3: retain most confident predictions
    zero_temp = model.state_dict()["modality_postprocessors.touch.1.log_logit_scale"].exp()
    probs = (logits / zero_temp).softmax(1) # probabilities
    # print("TTA Probabilities: " + str(probs))
    logits_filt, _, _ = confidence_filter(logits, probs, top=gamma, return_idx=True) # retain most confident views
    # print("TTA Filtered logits: " + str(logits_filt))
    # step4: zero temperature
    zero_temp = torch.finfo(logits_filt.dtype).eps
    # step5: marginalize
    p_bar = (logits_filt / zero_temp).softmax(1).sum(0)
    return p_bar

def classify_zero_tta(model, touch_images, class_labels):
    # class_labels = ["This is a touch image of " + x for x in classes]
    text_data=data.load_and_transform_text(class_labels, device=device)

    inputs = {
        "text": text_data
    }
    text_embeddings = model(inputs)["text"]

    norm_features = torch.nn.functional.normalize(text_embeddings)

    predictions = []

    for touch in touch_images:
        pred = zero(model, touch, norm_features, 32, 0.3)
        pred_index = pred.argmax()
        predictions.append(pred_index)
        print("TTA Predictions: " + str(pred))
    
    predicted_classes = [class_labels[pred_index] for pred_index in predictions]

    for c in predicted_classes:
        print(f"TTA Predicted class: {c}")

    return predicted_classes

def classify_tta_zero_single(model, inputs, classes):
    touch_image = inputs["touch"][0]
    inputs = {"text": inputs["text"]}
    text_embeddings = model(inputs)["text"]

    norm_features = torch.nn.functional.normalize(text_embeddings)

    pred = zero(model, touch_image, norm_features, 32, 0.3)
    # print(f"TTA Predictions: {pred}")
    pred_index = pred.argmax()

    return classes[pred_index]


if __name__ == "__main__":
    # checkpoint will be automatically downloaded
    model = x2touch(pretrained=True)
    model.eval()
    model = model.to(device)
    inputs = {}

    touch_paths=["./unitouch/wood.jpg"]
    touch = data.load_and_transform_vision_data(touch_paths, device=device)

    classes = ["wood","metal","grass", "rock","leather"]

    labels = [f'This feels like {x}' for x in classes]

    # classify_zero_tta(model, [touch], classes)

    class_label=data.load_and_transform_text(labels, device=device)
    inputs = {
        "touch": touch,
        "text": class_label
    }

    print("Predictd class: " + classify_tta_zero_single(model, inputs, classes))


    print("NON-TTA:")

    touch_embeddings = model(inputs)["touch"]
    text_embeddings = model(inputs)["text"]
    norm_features = torch.nn.functional.normalize(text_embeddings)
    similarity = touch_embeddings @ norm_features.T  # 形状: [2, 2]
    probs = torch.softmax(similarity, dim=-1)
    print("Probabilities: " + str(probs))
    print(f"Predicted class: {classes[probs.argmax()]}")