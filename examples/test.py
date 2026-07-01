import torch
import torch.nn.functional as F
from torchvision.utils import save_image

def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    print(f'[INFO] ratio={ratio}, resized_height={resized_height}, resized_width={resized_width}')
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    save_image(resized_img, "./examples/resize_temp.png")

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    print(f'[INFO] pad_height={pad_height}, pad_width={pad_width}')

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img

def main():
    # Pseudo RGB image in BCHW format: batch=1, channels=3, height=4, width=8.
    # It is wider than the target square, so resize_with_pad should pad on top.
    img = torch.zeros(1, 3, 480, 640)
    img[:, 0, :, :] = 1.0       # red background
    img[:, 1, int(480/4):int(480/4*3), int(480/8*2):int(480/8*6)] = 1.0   # green bar
    img[:, 2, :, 0:int(480/8)] = 1.0     # blue stripe on the left

    out = resize_with_pad(img, width=512, height=512, pad_value=-1)

    print("[INFO] input shape:", tuple(img.shape))
    print("[INFO] output shape:", tuple(out.shape))

    # Save normalized previews. The padded value -1 is mapped to 0 for visualization.
    save_image(img, "./examples/resize_input.png")
    save_image(out, "./examples/resize_output.png")

if __name__=='__main__':
    main()
