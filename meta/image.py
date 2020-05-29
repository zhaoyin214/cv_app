from interface.meta import Image, ImageSize

class ImageSizeNp(ImageSize):

    def __init__(self, image: Image):
        self._image = image

    @property
    def width(self) -> int:
        return self._image.shape[1]

    @property
    def height(self) -> int:
        return self._image.shape[0]

    @property
    def image(self) -> Image:
        return self._image
    @image.setter
    def image(self, image: Image):
        self._image = image


