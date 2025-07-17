**Canvas**
*Drawing Modes*
1.  freedraw (required here)
2. line, circle, rectangle...etc

*Key*
In Streamlit, every interactive widget or component (like sliders, text inputs, or custom components like a drawable canvas) can be given a unique key.
To uniquely identify the widget/component.
Prevents streamlit from confusing later on if we canvas is re-used

*Update Streamlit*


**Load model**
*@st.cache_resource:* used to enusre model is loaded only once even on re-runs
very useful for heavy resources ml models

**PreProcessing**

User is drawing digit on streamlit canvas, For the model to predict it correctly, the user_input need to be converted into mnist format, on which model is trained.

*MNIST Image Format (Target):*
Size: 28x28 pixels
Color: Grayscale
Foreground: White digit (pixel ~255)
Background: Black (pixel ~0)
Pixel values: Normalized to [0, 1]
Shape for model input: (1, 28, 28, 1) (for TensorFlow models)

