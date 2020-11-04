import time
import matplotlib.pyplot as plt
from load_data import *
from model import *

start_time = time.time()

with torch.no_grad():
    best_network = Network()
    best_network.cuda()
    best_network.load_state_dict(torch.load('./content/face_landmarks.pth'))
    best_network.eval()

    images, landmarks = next(iter(valid_loader))

    images = images.cuda()
    landmarks = (landmarks + 0.5) * 224

    predictions = (best_network(images).cpu() + 0.5) * 224
    predictions = predictions.view(-1, 68, 2)

    plt.figure(figsize=(10, 40))

    for img_num in range(8):
        plt.subplot(8, 1, img_num + 1)
        plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
        plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
        plt.scatter(landmarks[img_num, :, 0], landmarks[img_num, :, 1], c='g', s=5)

    plt.savefig('./content/result_30.jpg')

print('Total number of test images: {}'.format(len(valid_dataset)))

end_time = time.time()
print("Elapsed Time : {}".format(end_time - start_time))