# Face-Pix2Pix
Experimental project for restoring filtered face image using Pix2Pix

## 소개
pix2pix 모델을 사람 얼굴 이미지에 적용하기 위한 코드.

이미지에서 얼굴 부분을 인식해서 크롭 후 pix2pix 모델을 적용한다.
(학습단계에서도 동일하게 얼굴 인식 후 크롭한 후 train 진행함)

GPU사용이 불가능한 환경에서 이용할 수 있도록 colab에서도 실행 가능하도록 되어있음.

## 디렉토리 설정
1. 디렉토리는 업로드된 형태와 동일한 구조여야 함. 다만 data폴더는 경로 설정에 따라 바뀔 수 있음.
2. train 폴더는 디렉토리 구조와 폴더명까지 그대로 사용해야 함. a에는 target image(GT), b에는 filtered(마스킹된 또는 복원 이전의) 이미지가 들어가야 함. (*코드 실행 여부 확인을 위해 sample이미지는 아무렇게나 넣어두었음.*)
3. a, b 폴더 내의 pair 이미지의 파일명은 서로 일치해야 함.
4. 학습된 모델 가중치 .pt파일은 model_weight에 넣어야 함.
5. scripts 내의 .py파일들은 모두 같은 폴더에 있어야 함.

## 사용
0. dataset.py, model.py, optimizer.py 에서는 경로 설정 불필요
1. restore.py 에서 pt파일 경로, input이미지 경로, output이미지 경로가 지정되어야 함.
2. optimzer.py 에서 모델 학습 파라미터 조정
3. train.py 에서 train set 경로, pt파일 경로 설정 후 실행하여 모델 학습
4. restore.py 에서 model_weight 경로 설정 후 실행
5. 코랩 환경에서는 1~4의 단계 이후 restore.py 대신 cmd.ipynb을 실행할 것

## 이미지 파일 확장자
train 및 input 이미지로는 아래와 같은 형식을 지원함(PIL.open() 함수에서 지원하는 형식)

- BMP (Windows Bitmap)
- EPS (Encapsulated Postscript)
- GIF (Graphics Interchange Format)
- ICO (Windows Icon)
- JPEG (Joint Photographic Experts Group)
- PNG (Portable Network Graphics)
- PPM (Portable Pixmap)
- TIFF (Tagged Image File Format)
- WebP (Google WebP Image Format)

output 형식은 restore.py에서 코드 수정을 통해 지정 가능