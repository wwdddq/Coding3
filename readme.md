# Reality and Virtual World
Real-Time Image Style Transfer
[Open in Colab](https://colab.research.google.com/drive/1TrY03YGp5YR7jpOlBRLhhGvHfyEpgzaM)

## Introduction to the work
I hope that Reality and Virtual World will give the viewer a unique artistic experience and sensation. Through the rapid stylistic transfer of images in real time, the objective reality of photography is combined with the subjective expression of art, bringing a new sensory experience to the viewer. This technique blends elements of the real world with the artist's creativity, giving the photography a unique artistic and expressive quality.

The work uses a pre-trained [megenta model](https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization) by rchen152 and the TensorFlow Hub library to transform the images captured by the camera into images with a specified style in real time. With this creation I have attempted to push the boundaries of traditional photography and painting and to tightly integrate the real world with artistic expression.

## Design and Development
In designing this project, I was inspired by the work of Memo Akten [Learning to See](https://www.memo.tv/works/learning-to-see/). His work allows for real-time pre-processing of the camera input into the neural network, which already performs real-time inference. However, as my work computer is a Macbook, I was concerned that my GPU would not be sufficient to support me in my work concerning neural network generation, so I tried to find a fast style transfer method, combined with real-time camera input, to achieve a simple real-time style transfer.

In the Tenserflow Hub I found a tutorial on [Arbitrary image stylization](https://tensorflow.google.cn/hub/tutorials/tf2_arbitrary_image_stylization?hl=zh-cn), based on the model code in magenta. After working on this tutorial I tried to add a camera as the input content image and in the process I discovered that I could use the opencv function to call the local laptop camera directly in jupyter notebook, but not in Colab. So I started looking at how to use the local webcam in Colab.

```bash
# HTML code defining the video player used to capture the image
VIDEO_HTML = """
<video autoplay
 width=%d height=%d style='cursor: pointer;'></video>
<script>

var video = document.querySelector('video')

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream=> video.srcObject = stream)

var data = new Promise(resolve=>{
  video.onclick = ()=>{
    var canvas = document.createElement('canvas')
    var [w,h] =[video.offsetWidth, video.offsetHeight]
    canvas.width = w
    canvas.height = h
    canvas.getContext('2d')
          .drawImage(video, 0, 0, w, h)
    video.srcObject.getVideoTracks()[0].stop()
    video.replaceWith(canvas)
    resolve(canvas.toDataURL('image/jpeg', %f))
  }
})
</script>
"""

def take_photo(filename='photo.jpg', quality=1.0, size=(400,300)):
  display(HTML(VIDEO_HTML % (size[0],size[1],quality)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  f = io.BytesIO(binary)
  return np.asarray(Image.open(f))
```
As Colab is in the cloud server, Colab cannot call the local laptop camera directly with opencv. This code uses a method of acquiring camera images via JavaScript and HTML. A camera window is displayed using the display function and HTML code to allow the user to take a picture. Use JavaScript code to capture the camera image data and pass it to Python. Get the image data returned by the JavaScript code via eval_js("data"). Decode and convert the image data to a NumPy array. Return the NumPy array representing the captured image.


## Reference
- [Megenta model](https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization) 
- [Fast arbitrary image style transfer tutorial](https://tensorflow.google.cn/hub/tutorials/tf2_arbitrary_image_stylization?hl=zh-cn)
- [How to use camera in Colab](https://blog.csdn.net/weixin_42143481/article/details/105771183?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168681998016800225516119%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168681998016800225516119&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-105771183-null-null.142^v88^insert_down38v5,239^v2^insert_chatgpt&utm_term=colab%E6%91%84%E5%83%8F%E5%A4%B4&spm=1018.2226.3001.4187)
- [Thierry Feuz's art works](https://www.thierryfeuz.com/silent-winds/hth545k06wuypp2jrgdobw9cl2z1hy)
