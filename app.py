from flask import Flask ,render_template,request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

#load the model 
model = tf.saved_model.load("saved_model")

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

#the main page

@app.route("/",methods=["GET","POST"])
def home():
    
    prediction = ""
    image_path = ""
    top_predictions = []
    
    if request.method=="POST":
        
        #take the photo
        file=request.files["image"]
        #save the image
        image_path=os.path.join("static",file.filename)
        file.save(image_path)
        #reading the image
        img=image.load_img(image_path,target_size=(32,32))
         # تحويل الصورة إلى array
        img_array = image.img_to_array(img)

        # Normalization
        img_array = img_array / 255.0

        # إضافة بعد إضافي
        img_array = np.expand_dims(img_array, axis=0)

        # التوقع
        infer = model.signatures["serving_default"]
        prediction_result = infer(
             tf.constant(img_array)
        )
        prediction_result = list(prediction_result.values())[0].numpy()
        # استخراج أعلى 3 توقعات
        top_indices = prediction_result.argsort()[-3:][::-1]
        top_predictions = []
        for i in top_indices:
             label = class_names[i]
             confidence = float(prediction_result[i]) * 100
             top_predictions.append(
                 (label, round(confidence, 2))
            )
    #اول توقع         
    if len(top_predictions) > 0:
        prediction = top_predictions[0][0]
    else:
         prediction = "No prediction"
                     
                 
        
        
    return render_template(
        "index.html",
    prediction=prediction,
    top_predictions=top_predictions,
    image_path=image_path
    )    
    
# تشغيل الموقع
   

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)