import cv2
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision import transforms

IMAGE = 'people.jpg'

weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights)
model.eval()

preprocess = weights.transforms()
transform = transforms.ToTensor()

cv_image = cv2.imread(IMAGE)
rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
torch_image = transform(rgb_image)
batch = preprocess(torch_image).unsqueeze(0)

prediction = model(batch)[0]

labels = [weights.meta["categories"][i] for i in prediction["labels"]]

for i in range(len(labels)):
    score = prediction['scores'][i].item()
    if score > 0.3:
        x1, y1, x2, y2 = prediction['boxes'][i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = labels[i]
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(cv_image, f'{label}: {score:.4f}', (x1,y1+10), 1, .5, (255, 0 , 255), 1)

cv2.imshow('Result', cv_image)
cv2.waitKey(0)