import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_masks(image, mask, color, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.asarray(image)
    mask = mask > thresh

    color = np.asarray(color)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return out.astype(np.uint8)


def draw_boxes(img, bbox, names=None, identities=None, masks=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{:}{:d}'.format(names[i], id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        if masks is not None:
            mask = masks[i]
            img = draw_masks(img, mask, color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0]//4 + 3, y1 + t_size[1]//4 + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1]//4 + 3), cv2.FONT_HERSHEY_PLAIN, 0.5, [255, 255, 255], 1)
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
