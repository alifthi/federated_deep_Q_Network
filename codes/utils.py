import cv2 as cv
from config import STATE_SIZE
class utils:
    def __init__(self) -> None:
        self.state_size=STATE_SIZE
    def preprocessing(self,image):
        image=cv.resize(image,self.state_size[:-1])
        image=image/255.0
        return image