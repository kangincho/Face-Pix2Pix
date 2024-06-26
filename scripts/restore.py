import os
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import face_recognition
from model import GeneratorUNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 설정
path2weights_gen = '/content/drive/MyDrive/pix2pix_backup/model_weight/weights_gen.pt'  # pt 파일 저장된 경로
path2input = '/content/drive/MyDrive/pix2pix_backup/data/input_img'  # 원본 input 이미지가 저장된 폴더 경로
path2output = '/content/drive/MyDrive/pix2pix_backup/data/output_img'  # output 이미지를 저장할 폴더 경로

# 가중치 불러오기
model_gen = GeneratorUNet().to(device)
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Resize((256, 256))
])

class NewDataset(Dataset):
    def __init__(self, path2input, transform=None):
        self.path2input = path2input
        self.img_filenames = [x for x in listdir(path2input)]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(join(self.path2input, self.img_filenames[index])).convert('RGB')

        # 이미지에서 얼굴 인식 및 크롭
        img_array = face_recognition.load_image_file(join(self.path2input, self.img_filenames[index]))
        face_locations = face_recognition.face_locations(img_array)

        if len(face_locations) > 0:
            # 발견된 첫 번째 얼굴에 대해서만 처리
            top, right, bottom, left = face_locations[0]
            img_cropped = img.crop((left, top, right, bottom))

            if self.transform:
                img_cropped = self.transform(img_cropped)

            return img_cropped, self.img_filenames[index]  # 파일명도 반환
        else:
            # 얼굴이 감지되지 않은 경우 원본 이미지 반환
            if self.transform:
                img = self.transform(img)

            return img, self.img_filenames[index]  # 파일명도 반환

    def __len__(self):
        return len(self.img_filenames)

# 데이터셋 불러오기
new_ds = NewDataset(path2input, transform=transform)

# 결과 저장 폴더 생성
os.makedirs(path2output, exist_ok=True)

# evaluation model
model_gen.eval()

# 이미지 변환 및 저장
with torch.no_grad():
    for img, filename in new_ds:
        img = img.unsqueeze(0).to(device)

        # 모델을 통해 이미지 변환
        fake_img = model_gen(img).detach().cpu().squeeze()

        # 정규화 해제
        fake_img = (fake_img * 0.5) + 0.5

        # PIL 이미지로 바꾸기
        fake_img = transforms.ToPILImage()(fake_img)

        # 파일명에서 확장자 제거.
        filename_without_ext = os.path.splitext(filename)[0]

        # 이미지 저장
        output_path = join(path2output, f"restored_{filename_without_ext}.png")
        fake_img.save(output_path, "PNG")

# jpeg로 저장되기를 원하지 않으면 아래처럼 수정
'''
with torch.no_grad():
    for img, filename in new_ds:
        img = img.unsqueeze(0).to(device)

        # 모델을 통해 이미지 변환
        fake_img = model_gen(img).detach().cpu().squeeze()

        # 정규화 해제
        fake_img = (fake_img * 0.5) + 0.5

        # PIL 이미지로 바꾸기
        fake_img = transforms.ToPILImage()(fake_img)

        # 이미지 저장
        output_path = join(path2output, f"restored_{filename}")
        fake_img.save(output_path)
        '''


