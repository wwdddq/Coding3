# Reality and Virtual World
Real-Time Image Style Transfer
![Coding3](https://github.com/wwdddq/Coding3/blob/main/images/result.png)

[Open in Colab](https://colab.research.google.com/drive/1TrY03YGp5YR7jpOlBRLhhGvHfyEpgzaM)

[Presentation Video]()


## Introduction
I hope that Reality and Virtual World will give the viewer a unique artistic experience and sensation. Through the rapid stylistic transfer of images in real time, the objective reality of photography is combined with the subjective expression of art, bringing a new sensory experience to the viewer. This technique blends elements of the real world with the artist's creativity, giving the photography a unique artistic and expressive quality.

The work uses a pre-trained [megenta model](https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization) by rchen152 and the TensorFlow Hub library to transform the images captured by the camera into images with a specified style in real time. With this creation I have attempted to push the boundaries of traditional photography and painting and to tightly integrate the real world with artistic expression.


## Design and Development
In designing this project, I was inspired by the work of Memo Akten [Learning to See](https://www.memo.tv/works/learning-to-see/). His work allows for real-time pre-processing of the camera input into the neural network, which already performs real-time inference. However, as my work computer is a Macbook, I was concerned that my GPU would not be sufficient to support me in my work concerning neural network generation, so I tried to find a fast style transfer method, combined with real-time camera input, to achieve a simple real-time style transfer.

In the Tenserflow Hub I found a tutorial on [Arbitrary image stylization](https://tensorflow.google.cn/hub/tutorials/tf2_arbitrary_image_stylization?hl=zh-cn), based on the model code in magenta. 

In the code, I replaced the original stylised image with one by my favourite artist Thierry Feuz [Silent Winds Avalon](https://www.thierryfeuz.com/silent-winds/hth545k06wuypp2jrgdobw9cl2z1hy). His work is gorgeous, disorienting and fantastically broken, like something out of this world or a cellular nerve. It's worth noting that users can customise the url of the style image in the load example images to explore the combination of reality and the virtual in their own personal aesthetic.

```bash
# @title Load example images  { display-mode: "form" }

# style image can be customized
style_image_url = 'https://images.squarespace-cdn.com/content/v1/5881f213a5790ac16d505983/1486549429319-3BHSRUYJO1TSUS5R2JEM/Silent-Winds-Avalon%2C-110x90cm%2C-2014-copie.jpg?format=1000w'  # @param {type:"string"}
output_image_size = 384  # @param {type:"integer"}

# The content image size can be arbitrary.
content_img_size = (output_image_size, output_image_size)
# The style prediction model was trained with image size 256 and it's the
# recommended image size for the style image (though, other sizes work as
# well but will lead to different results).
style_img_size = (256, 256)

style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
show_n([style_image], ['Style image'])
```

After working on this tutorial I tried to add a camera as the input content image and in the process I discovered that I could use the opencv function to call the local laptop camera directly in jupyter notebook, but not in Colab. So I started looking at how to use the local webcam in Colab. I would like to thank [静心定心](https://blog.csdn.net/weixin_42143481/article/details/105771183?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168681998016800225516119%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168681998016800225516119&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-105771183-null-null.142^v88^insert_down38v5,239^v2^insert_chatgpt&utm_term=colab%E6%91%84%E5%83%8F%E5%A4%B4&spm=1018.2226.3001.4187) for sharing and successfully solving this problem.

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

# Define functions for taking photos
def take_photo(filename='photo.jpg', quality=1.0, size=(400,300)):
  display(HTML(VIDEO_HTML % (size[0],size[1],quality)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  f = io.BytesIO(binary)
  return np.asarray(Image.open(f))
```
As Colab is in the cloud server, Colab cannot call the local laptop camera directly with opencv. This code uses a method of acquiring camera images via JavaScript and HTML. A camera window is displayed using the display function and HTML code to allow the user to take a picture. Use JavaScript code to capture the camera image data and pass it to Python. Get the image data returned by the JavaScript code via eval_js("data"). Decode and convert the image data to a NumPy array. Return the NumPy array representing the captured image.

Next I created a folder on Google Drive to save the captured images with the stylised images.
```bash
# Create a folder for saving images
save_folder = '/content/drive/MyDrive'  # Specify the path of the saved folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
```

The following code is the core part of this project and implements the following functions:
1. Turn on the camera and start capturing a video stream.
2. In an infinite loop, images are captured through the camera and image stylisation is performed.
3. The captured and stylised images are saved to a file.
4. After each loop, the counter is incremented and used to generate the file name.
5. The user can exit the loop by pressing the 'q' key.
6. At the end of the loop, the camera resource is released and the window is closed.

In summary, this code allows the user to apply image style transitions to the camera's video stream in real time and save the captured and stylised images.

```bash
# Turn on the camera
cap=cv2.VideoCapture(0)

# Defining the counter
count = 0

while True:
    # Capture images
    content_img = take_photo()

    # Converting images to float32 arrays
    content_img = content_img.astype(np.float32) / 255.0

    # Image resizing and dimensionality
    content_img = tf.image.resize(content_img, content_img_size)
    content_img = tf.expand_dims(content_img, axis=0)

    # Image style conversion
    outputs = hub_module(content_img, tf.constant(style_image))
    stylized_image = outputs[0]

    # Display images
    plt.imshow(stylized_image[0])
    plt.axis('off')
    plt.show()

    # Save captured images and stylized images
    content_img_array = np.squeeze(content_img.numpy())  # Converting TensorFlow tensors to NumPy arrays
    stylized_img_array = np.squeeze(stylized_image.numpy())

    cv2.imwrite(os.path.join(save_folder, f'content_img_{count}.jpg'), content_img_array)
    cv2.imwrite(os.path.join(save_folder, f'stylized_image_{count}.jpg'), stylized_img_array)


    # Add counter
    count += 1

    # Check for key input
    try:
        key = getpass(prompt='Press q to quit:')
        if key == 'q':
            break
        clear_output(wait=True)
    except KeyboardInterrupt:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
```


## Summary and Reflection
In this project, I implemented style conversion of camera images using the Python programming language and related libraries. Through learning and experimentation, I successfully loaded and applied a pre-trained image style conversion model and implemented real-time image capture and conversion. This project has improved my understanding of image processing and model applications, and honed my programming and problem solving skills.

In future projects, I hope to develop this image style conversion functionality into a mobile application. Through a user-friendly interface, users can easily select a style image, take or import a content image and preview the style conversion effect in real time. By developing this project into an app, I hope to make image style conversion easy to use for more people and provide them with a fun and creative interactive tool. At the same time, I also want to try to train a model based on the magenta model to match my own aesthetic.


## Reference
- [Megenta model](https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization) 
- [Fast arbitrary image style transfer tutorial](https://tensorflow.google.cn/hub/tutorials/tf2_arbitrary_image_stylization?hl=zh-cn)
- [How to use camera in Colab](https://blog.csdn.net/weixin_42143481/article/details/105771183?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168681998016800225516119%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168681998016800225516119&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-105771183-null-null.142^v88^insert_down38v5,239^v2^insert_chatgpt&utm_term=colab%E6%91%84%E5%83%8F%E5%A4%B4&spm=1018.2226.3001.4187)
- [Thierry Feuz's art works](https://www.thierryfeuz.com/silent-winds/hth545k06wuypp2jrgdobw9cl2z1hy)
