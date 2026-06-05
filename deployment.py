import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import traceback

# Title and description
st.title("Car vs Bike Classifier")

st.write(
    "Upload an image of a vehicle and click Predict."
)

st.warning(
    "Disclaimer: This classifier is trained exclusively on images of Cars and Bikes. "
    "Uploading out-of-distribution images (such as text, logos, or other objects) will force the model "
    "to classify them as either a Car or a Bike, which may produce incorrect high-confidence results."
)

@st.cache_resource
def load_model():
    """
    Loads the Keras MobileNetV2 model once and caches it.
    
    Why models are saved as .keras files:
    - The '.keras' format is the standard Keras native format. It is a single-file, 
      zipped archive containing the model architecture configuration, weights, and 
      compilation metadata. It is self-contained and highly portable compared to 
      legacy formats like H5 or SavedModel folders.
      
    Why loading once is important:
    - Deep learning models can be large (e.g., tens or hundreds of megabytes) and 
      contain millions of parameters. Loading the model takes significant time 
      (disk I/O) and CPU/GPU memory initialization. Doing this on every user action 
      would make the app extremely laggy.
      
    Why Streamlit caching (@st.cache_resource) improves performance:
    - Streamlit runs the script from top to bottom on every user interaction (e.g., clicks). 
      Caching allows Streamlit to store the loaded model object in memory and reuse 
      it across runs instead of re-reading it from disk and re-instantiating the model, 
      saving both CPU cycles and memory space.
    """
    model_path = "model/mobilenetv2_car_bike.keras"
    model = tf.keras.models.load_model(model_path)
    
    # Print the model summary in terminal/logs for verification
    print("\n--- Verification: Loaded Model Summary ---")
    model.summary()
    print("-------------------------------------------\n")
    
    return model

# 1. EXCEPTION SCENARIO: Model Loading Failures
# We wrap the model loading in a try-except block. If loading fails (e.g., missing file, 
# incompatible TF version, file corruption), we show a clean user-facing error and print
# the full system traceback to stderr for developers.
try:
    model = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error("Failed to load the classification model. Please contact support or check that the model file exists.")
    print(f"\n[TECHNICAL ERROR] Model Loading Failed:\n{traceback.format_exc()}\n")
    st.stop()

def preprocess_image(image):
    """
    Preprocesses the input image for the CNN model inference.
    
    Steps:
    1. Convert image to RGB format to discard alpha channels or convert grayscale to 3 channels.
    2. Resize the image to 224x224 pixels as required by the MobileNetV2/CNN model input layer.
    3. Convert the PIL Image into a NumPy array for tensor operations.
    4. Normalize pixel values from [0, 255] range to [0.0, 1.0] by dividing by 255.0.
    5. Add a batch dimension to transform shape from (224, 224, 3) to (1, 224, 224, 3).
    """
    # 1. Convert image to RGB format (important for PNGs with transparency or Grayscale images)
    rgb_image = image.convert("RGB")
    
    # 2. Resize to 224x224
    resized_image = rgb_image.resize((224, 224))
    
    # 3. Convert PIL Image to a NumPy array
    img_array = np.array(resized_image)
    
    # 4. Normalize pixel values to [0.0, 1.0]
    normalized_image = img_array / 255.0
    
    # 5. Add a batch dimension (from (224, 224, 3) to (1, 224, 224, 3))
    batched_image = np.expand_dims(normalized_image, axis=0)
    
    return batched_image

def predict_image(model, processed_image):
    """
    Runs model inference on the preprocessed image.
    
    Inference Details:
    - Passes the 4D preprocessed image array of shape (1, 224, 224, 3) to the model.
    - Uses model.predict() to get the sigmoid output.
    
    Sigmoid Interpretation & Decision Making:
    - The output activation is a sigmoid function, outputting a value in [0, 1].
    - This output represents the probability of the positive class (Car = 1).
    - Thresholding: We use 0.5 as the decision boundary.
      - If raw value > 0.5, predicted class is 'Car'. The confidence is the raw value.
      - If raw value <= 0.5, predicted class is 'Bike'. The confidence is (1 - raw value).
    """
    # 1. Run inference using model.predict
    prediction = model.predict(processed_image)
    
    # 2. Store the prediction output (extract the raw value from the (1, 1) result matrix)
    raw_val = float(prediction[0][0])
    
    # 3. Decision logic using thresholding at 0.5
    if raw_val > 0.5:
        predicted_class = "Car"
        confidence = raw_val
    else:
        predicted_class = "Bike"
        confidence = 1.0 - raw_val
        
    return predicted_class, confidence, raw_val

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # 2. EXCEPTION SCENARIO: Invalid / Corrupted Images
    # We load and call .load() on the PIL Image in a try-except. If the file is not an image
    # or is corrupted, PIL will fail here. We print details to logs and stop streamlit runner.
    try:
        image = Image.open(uploaded_file)
        image.load()  # Force loading of data to catch decompression/corruption errors
    except Exception as e:
        st.error("The uploaded image is invalid or corrupted. Please upload a valid JPG, JPEG, or PNG file.")
        print(f"\n[TECHNICAL ERROR] Image verification failed:\n{traceback.format_exc()}\n")
        st.stop()

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )
    
    # 3. EXCEPTION SCENARIO: Image Preprocessing Failures
    # We wrap preprocessing in try-except in case of format/shape compatibility issues.
    try:
        processed_image = preprocess_image(image)
    except Exception as e:
        st.error("Image preprocessing failed. The image properties might not be supported.")
        print(f"\n[TECHNICAL ERROR] Preprocessing failed:\n{traceback.format_exc()}\n")
        st.stop()
    
    # Display the resulting shape using Streamlit
    st.write(f"**Processed image shape:** `{processed_image.shape}`")

    if st.button("Predict"):
        
        # 4. EXCEPTION SCENARIO: Prediction Failures
        # We wrap model prediction execution in try-except in case tensor shapes are incorrect,
        # resources are exhausted, or Keras crashes.
        try:
            predicted_class, confidence, raw_val = predict_image(model, processed_image)
        except Exception as e:
            st.error("Prediction failed. An error occurred inside the model network.")
            print(f"\n[TECHNICAL ERROR] Model Prediction failed:\n{traceback.format_exc()}\n")
            st.stop()
        
        # Result Display Section
        st.write("---")
        st.subheader("Classification Result")
        
        # Define uncertainty: if confidence lies between 40% and 60% (since confidence is >= 50%
        # for predicted class, this checks if the prediction is close to decision boundary i.e., <= 60% confidence).
        # We also check if raw sigmoid output falls in [0.40, 0.60].
        is_uncertain = (0.40 <= raw_val <= 0.60) or (0.40 <= confidence <= 0.60)
        
        class_emoji = "🚗" if predicted_class == "Car" else "🚲"
        
        # Use st.success for high-confidence and st.warning for uncertain predictions
        result_msg = f"**Prediction:** {class_emoji} {predicted_class}  \n**Confidence:** {confidence:.2%}"
        if is_uncertain:
            st.warning(f"**Uncertain Prediction**  \n{result_msg}")
        else:
            st.success(f"**High-Confidence Prediction**  \n{result_msg}")
            
        # Display clean Prediction Cards
        st.markdown(f"""
        <div style="padding: 16px; border-radius: 8px; background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid {'#FFA500' if is_uncertain else '#4CAF50'}; margin-bottom: 20px;">
            <span style="font-size: 14px; text-transform: uppercase; color: #888; font-weight: bold;">Prediction Card</span>
            <div style="font-size: 24px; font-weight: bold; margin-top: 8px; color: #fff;">
                {class_emoji} {predicted_class}
            </div>
            <div style="font-size: 16px; color: #bbb; margin-top: 4px;">
                Confidence Score: <b>{confidence:.2%}</b>
            </div>
            <div style="font-size: 14px; color: #888; margin-top: 4px;">
                Raw Sigmoid Output: <code>{raw_val:.4f}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed Probability Breakdowns
        car_prob = raw_val * 100
        bike_prob = (1.0 - raw_val) * 100
        
        st.write("####Probability Interpretation Breakdown")
        st.write(f"- **Car Probability:** `{car_prob:.2f}%`")
        st.write(f"- **Bike Probability:** `{bike_prob:.2f}%`")
        
        # Explanation of terms
        st.write("""
        > **Explanatory Glossary:**
        > - **Confidence Score:** The probability of the predicted class (Car probability if Car is predicted, Bike probability if Bike is predicted).
        > - **Thresholding:** We use a threshold of `0.5` on the sigmoid output. Any value $> 0.5$ is classified as a Car; otherwise, it is a Bike.
        > - **Sigmoid Interpretation:** The model final neuron maps features to a single probability output. Sigmoid outputs close to `1.0` or `0.0` indicate strong model certainty.
        """)
        
        # Display the temporary debug section showing outputs
        st.write("### 🛠&nbsp; Temporary Debug Section")
        st.write(f"- **Raw Prediction Value:** `{raw_val:.6f}`")
        st.write(f"- **Interpreted Class:** `{predicted_class}`")
        st.write(f"- **Confidence:** `{confidence:.2%}`")

# Explain why resizing, normalization, and batch dimensions are required for CNN inference
st.info("""
### Why are these preprocessing steps required for CNN inference?

1. **Resizing ($224 \\times 224$):**
   Convolutional Neural Networks (CNNs) have fixed-size input layers (e.g., $224 \\times 224 \\times 3$) because they are connected to fully-connected dense layers at the end. These dense layers have a set number of input neurons which requires the spatial dimensions of the input tensor to be exactly consistent.

2. **Normalization (dividing by 255.0):**
   Raw images have pixel values between `0` and `255`. Normalizing these values to `[0.0, 1.0]` scales the values down, which prevents gradient explosion, improves gradient descent stability, matches the training data scale, and helps the model converge faster during inference.

3. **Batch Dimension (shape `(1, 224, 224, 3)`):**
   Deep learning frameworks (like TensorFlow/Keras or PyTorch) perform operations in batches to leverage GPU parallelism. Therefore, even for a single image prediction, the model expects a 4D tensor of shape `(batch_size, height, width, channels)`. Adding the batch dimension (making the shape `(1, 224, 224, 3)`) satisfies this tensor format constraint.
""")

# Explain sigmoid activation and outputs
st.info("""
###Sigmoid Activation & Binary Classification

1. **Why the output is a single number:**
   The output layer of our model has a single neuron (`Dense(1)`). In binary classification (Car vs. Bike), we only need to output a single value representing the probability of the positive class ($P(\text{Class} = \text{Car})$). The probability of the negative class ($P(\text{Class} = \text{Bike})$) is simply $1 - P(\text{Class} = \text{Car})$, so a single unit is sufficient.

2. **Why the output lies between 0 and 1:**
   The output neuron uses the **Sigmoid** activation function:
   $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$
   No matter how large or small the input ($z$) is, the mathematical range of $\\sigma(z)$ is bound between $0$ and $1$. This makes the output ideal for interpretation as a probability.

3. **Shape and Meaning of the Output:**
   - **Output Shape:** `(1, 1)` (1 sample batch, 1 classification output value).
   - **Meaning:** The value represents the probability that the uploaded image is a **Car**.
""")